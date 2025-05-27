import os
import torch
import torch.nn as nn
from transformers import BertModel
from TTA import FinalDecoder, TTA


class MultiHeadAttention(nn.Module):
    def __init__(self,d_self,d_ref,n_head,bias=True):
        super(MultiHeadAttention,self).__init__()
        self.d_self=d_self
        self.n_head=n_head
        self.head_dim=d_ref//n_head
        self.scale=self.head_dim**-0.5
        self.norm_q=nn.LayerNorm(d_self)
        self.norm_k=nn.LayerNorm(d_ref)
        self.norm_v=nn.LayerNorm(d_ref)
        self.to_query=nn.Linear(d_self,d_ref,bias=bias)
        self.to_key=nn.Linear(d_ref,d_ref,bias=bias)
        self.to_value=nn.Linear(d_ref,d_ref,bias=bias)
    def forward(self,q,k,v):
        b,n,d=q.shape
        _,l,bert_dim=k.shape
        #[b, n, n_head, head_dim]
        query=self.to_query(self.norm_q(q)).view((b,n,self.n_head,self.head_dim))
        #[b, l, n_head, head_dim]
        key=self.to_key(self.norm_k(k)).view((b,l,self.n_head,self.head_dim))
        #[b, l, n_head, head_dim]
        value=self.to_value(self.norm_v(v)).view((b,l,self.n_head,self.head_dim))
        #[b,n,l,n_head]
        alpha=torch.einsum("bnhd,blhd->bnlh",query,key)*self.scale
        alpha=alpha.softmax(dim=2)
        return alpha,value
class GlobalAttention(nn.Module):
    def __init__(self,l_u,d_self=768,d_ref_bert=768,bias=True):
        super(GlobalAttention,self).__init__()
        self.to_q_beta=nn.Linear(in_features=d_self,out_features=d_self,bias=bias)
        self.to_k_beta=nn.Linear(in_features=l_u*d_ref_bert,out_features=d_self,bias=bias)
        self.active=nn.Sigmoid()
    def forward(self,p,u):
        b,l_u,d=u.shape
        u = u.reshape(b, -1)
        u=self.to_k_beta(u).unsqueeze(dim=1)
        p=self.to_q_beta(p)
        beta=torch.einsum("bnd,bld->bnl",p,u)
        beta=self.active(beta)
        return beta
class GlobalLocalMultiHeadAttention(nn.Module):
    def __init__(self,l_u,d_self,d_ref_bert=768,n_head=8,bias=True):
        super(GlobalLocalMultiHeadAttention,self).__init__()
        self.n_head=n_head
        self.l_u=l_u
        self.attention_utterance=MultiHeadAttention(d_self,d_ref_bert,n_head,bias=bias)
        self.attention_global=GlobalAttention(l_u,d_self,d_ref_bert,bias=bias)
    def forward(self,p,u):
        #v_g:[b,l_d,n_head,head_dim]
        alpha_u,v_u=self.attention_utterance(p,u,u)
        beta=self.attention_global(p,u)
        b,n,_,_=alpha_u.shape
        beta_u=beta.repeat((1,1,self.n_head*self.l_u)).view((b,n,self.l_u,self.n_head))
        gamma_u=torch.mul(alpha_u,beta_u)
        C_U_att=torch.einsum("bnlh,blhd->bnhd",gamma_u,v_u).reshape((b,n,-1))
        output=torch.cat((p,C_U_att),dim=-1)
        return output
class DenseResult(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenseResult,self).__init__()
        self.fn1=nn.Linear(input_dim,input_dim//2)
        self.fn2=nn.Linear(input_dim//2,output_dim)
    def forward(self,x):
        x=self.fn1(x)
        x=self.fn2(x)
        return x
class  AGLCF_TTA(nn.Module):
    def __init__(self,
                 intent_shift_size,
                 intent_size,
                 slots_size,
                 max_len,
                 l_u,# length of previous utterances
                 l,
                 d_self=768,#dimension of the current utterance
                 hidden_size=256,
                 d_ref_bert=768,#the embedding dimension of the previous utterances
                 n_head=8,
                 bias=True,
                 pretrained_model="bert-base-uncased",):
        super(AGLCF_TTA,self).__init__()
        self.bert=BertModel.from_pretrained(pretrained_model)
        self.globalLocal_multi_head_attention=GlobalLocalMultiHeadAttention(l_u=l_u,d_self=d_self,d_ref_bert=d_ref_bert,n_head=n_head,bias=bias)
        self.intent_bi_lstm=nn.LSTM(input_size=(d_self+d_ref_bert),hidden_size=hidden_size,bidirectional=True)
        self.slots_bi_lstm=nn.LSTM(input_size=(d_self+d_ref_bert),hidden_size=hidden_size,bidirectional=True)
        self.final_decoder=FinalDecoder(hidden_size*2,hidden_size*2,max_len,intent_shift_size,intent_size,slots_size)
        self.tta=TTA(hidden_size*2, d_ref_bert,intent_size, slots_size,l)
    def forward(self,d,u):
        p=self.bert(d["text"],d["attention_masks"]).last_hidden_state
        p_context=self.globalLocal_multi_head_attention(p,u)
        intent_embedding, (_, _) = self.intent_bi_lstm(p_context)
        slot_embedding, (_, _) = self.slots_bi_lstm(p_context)
        intent_embedding,slot_embedding=self.tta(intent_embedding,slot_embedding,u)
        intent_shift_result,intent_result, slots_result=self.final_decoder(intent_embedding,slot_embedding)
        return intent_shift_result,intent_result,slots_result
