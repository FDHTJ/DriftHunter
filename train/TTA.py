import torch
import torch.nn as nn
import torch.nn.functional as F



class UtteranceAttention(nn.Module):
    def __init__(self,bert_dim,meta_intent_number):
        super(UtteranceAttention,self).__init__()
        self.w_l2 = nn.Parameter(torch.empty((meta_intent_number,bert_dim)),requires_grad=True)
        nn.init.xavier_uniform_(self.w_l2)
        self.w_l1=nn.Linear(bert_dim,bert_dim)
        self.tenh=nn.Tanh()
        self.h_w_for_label_attention=nn.Parameter(torch.empty((meta_intent_number,bert_dim)),requires_grad=True)
        nn.init.xavier_uniform_(self.h_w_for_label_attention)
        self.w3=nn.Linear(bert_dim,bert_dim)
        self.w4=nn.Linear(bert_dim,bert_dim)
        self.sigmoid=nn.Sigmoid()
    def adaptive_fusion(self,H1,H2):
        alpha=(self.sigmoid(self.w3(H1)))/(self.sigmoid(self.w3(H1))+self.sigmoid(self.w4(H2)))
        return alpha
    def forward(self,p):
        h1=self.tenh(self.w_l1(p))
        h1=h1.transpose(1,2)
        h1=torch.matmul(self.w_l2,h1)
        h1=h1.softmax(dim=-1)
        H1=torch.matmul(h1,p)
        h2=torch.matmul(self.h_w_for_label_attention,p.transpose(1,2))
        H2=torch.matmul(h2,p)
        alpha=self.adaptive_fusion(H1,H2)
        H=alpha*H1+(1-alpha)*H2
        return H
class TTADecoder(nn.Module):
    def __init__(self,bert_dim,utterance_embedding):
        super(TTADecoder,self).__init__()
        self.w0=nn.Linear(bert_dim+utterance_embedding,bert_dim)
        self.w1 = nn.Linear(bert_dim * 3, bert_dim)
        self.active = nn.Sigmoid()
        self.conv = nn.Conv1d(in_channels=bert_dim, out_channels=bert_dim, kernel_size=3, padding=1, padding_mode='circular')
        self.w2 = nn.Linear(bert_dim, bert_dim)
        self.intent_fn=nn.Linear(bert_dim, bert_dim)
        self.slot_fn=nn.Linear(bert_dim, bert_dim)
    def forward(self,p,H):
        H=self.w0(H)
        p = torch.matmul(torch.matmul(p, H.transpose(1, 2)).softmax(dim=-1), H) + p
        p_transposed = p.transpose(1, 2)
        p_conv = self.conv(p_transposed)
        p_conv = p_conv.transpose(1, 2)
        p1 = self.w2(self.active(p_conv)) + p
        intent_embedding=self.intent_fn(p1)
        slot_embedding=self.slot_fn(p1)
        return intent_embedding,slot_embedding
class TTA(nn.Module):
    def __init__(self,bert_dim,utterance_embedding,intent_number,slots_number,l=0.2):
        super(TTA,self).__init__()
        self.utteranceAttention=UtteranceAttention(bert_dim*2,intent_number+slots_number)
        self.transition_matrix=torch.zeros((intent_number+slots_number,intent_number+slots_number))
        nn.init.xavier_uniform_(self.transition_matrix)
        self.global_attention=nn.Parameter(torch.empty((intent_number+slots_number,utterance_embedding)),requires_grad=True)
        nn.init.xavier_uniform_(self.global_attention)
        self.last_result=torch.empty((intent_number+slots_number,bert_dim))
        nn.init.xavier_uniform_(self.last_result)
        self.activate=nn.ReLU()
        self.local_fn=nn.Linear(bert_dim*2,bert_dim)
        self.global_fn=nn.Linear(utterance_embedding,utterance_embedding)
        self.tta_intent_decoder=TTADecoder(bert_dim,utterance_embedding)
        self.tta_slot_decoder=TTADecoder(bert_dim,utterance_embedding)
        self.l=l
    def update(self,curr):
        curr=curr.mean(dim=0)
        similarity_matrix = F.cosine_similarity(
            self.last_result.unsqueeze(1).to(curr.device),  # 形状变为 (8, 1, 768)
            curr.unsqueeze(0),  # 形状变为 (1, 8, 768)
            dim=-1  # 在最后一个维度（768）计算相似度
        )

        # noise = torch.randn_like(similarity_matrix) * 0.1
        # similarity_matrix += noise
        # logits = similarity_matrix  # 任意形状 (batch, N)
        # similarity_matrix = entmax15(similarity_matrix, dim=-1)  # 或 entmax15
        similarity_matrix=similarity_matrix.softmax(dim=-1)
        l=self.l
        self.transition_matrix=l*self.transition_matrix.to(curr.device)+similarity_matrix.detach()*(1-l)
        self.last_result=curr.detach()#0.5*self.last_result.to(curr.device)+ 0.5*
    def forward(self,intent_embedding,slots_embedding,u):
        intent_original,slots_original=intent_embedding,slots_embedding
        joint_embedding_original=torch.cat((intent_original,slots_original),dim=-1)
        joint_embedding=self.utteranceAttention(joint_embedding_original)
        self.transition_matrix=self.transition_matrix.to(slots_embedding.device)
        matrix=self.transition_matrix
        matrix=matrix.transpose(0,1)
        joint_embedding= self.local_fn(torch.matmul(matrix,joint_embedding))
        self.update(joint_embedding)
        global_information= self.global_fn(torch.matmul(self.transition_matrix.transpose(0,1),torch.matmul(torch.matmul(self.global_attention,u.transpose(1,2)),u)))
        H=torch.cat((joint_embedding,global_information),dim=-1)
        intent_enhance_embedding1,slot_enhance_embedding1=self.tta_intent_decoder(intent_original,H)
        intent_enhance_embedding2,slot_enhance_embedding2=self.tta_slot_decoder(slots_original,H)
        intent_enhance_embedding=intent_enhance_embedding1+intent_enhance_embedding2
        slot_enhance_embedding=slot_enhance_embedding1+slot_enhance_embedding2
        return intent_enhance_embedding,slot_enhance_embedding
class FinalDecoder(nn.Module):
    def __init__(self,intent_dim,slots_dim,max_len,intent_shift_size,intent_size,slots_size):
        super(FinalDecoder,self).__init__()
        self.fn_intent_shift = nn.Linear(in_features=intent_dim, out_features=intent_shift_size)
        self.fn_intent = nn.Linear(in_features=intent_dim, out_features=intent_size)
        self.w_intent = nn.Parameter(torch.empty((1, max_len)))
        nn.init.xavier_uniform_(self.w_intent)
        self.w_intent_shift = nn.Parameter(torch.empty((1, max_len)))
        nn.init.xavier_uniform_(self.w_intent_shift)
        self.fn_slots = nn.Linear(in_features=slots_dim, out_features=slots_size)
    def forward(self,p_intent_embedding,p_slots_embedding):
        p_intent = torch.matmul(self.w_intent_shift, p_intent_embedding)
        p_intent_shift = torch.matmul(self.w_intent, p_intent_embedding)
        intent = self.fn_intent(p_intent).squeeze(dim=1)
        intent_shift=self.fn_intent_shift(p_intent_shift).squeeze(dim=1)
        slots = self.fn_slots(p_slots_embedding)
        return intent_shift,intent, slots