# Intent Drift Detection in Continuous Conversation via Temporal Transition Accumulating(TTA)
All the dataset we used can be obtain by follows：

[ATIS(Airline Travel Information System)](https://github.com/howl-anderson/ATIS_dataset/)

[SIM(The Simulated-Dialogue dataset)](https://github.com/google-research-datasets/simulated-dialogue)

[Multiwoz(Multi-Domain Wizard-of-Oz Datase)](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)

Because the processed Multiwoz is too large to upload, so we only keep the processed ATIS and SIM for our training and testing.

But we provide the code to process the data in /data/process_data.py, before you try  to get the processed data, merge all the dialogues in the orignal dataset is recommended.

The details of data process is represented in the /data/README.md.

The hyperparemeters of α, β, and γ are as follows：

Table: Hyperparameter on ATIS：
| Method             | α    | β    | γ    |
| ----------------- | ---- | ---- | ---- |
| BERT-TTA          | 0.05 | 0.15 | 0.80 |
| RNNContextual-TTA | 0.05 | 0.25 | 0.70 |
| AGLCF-TTA         | 0.05 | 0.25 | 0.70 |
| DHLG-TTA          | 0.10 | 0.20 | 0.70 |

Table: Hyperparameter on SIM：
| Method             | α    | β    | γ    |
| ----------------- | ---- | ---- | ---- |
| BERT-TTA          | 0.05 | 0.15 | 0.80 |
| RNNContextual-TTA | 0.05 | 0.25 | 0.70 |
| AGLCF-TTA         | 0.05 | 0.25 | 0.70 |
| DHLG-TTA          | 0.10 | 0.30 | 0.60 |

Table: Hyperparameter on Multiwoz：
| Method             | α    | β    | γ    |
| ----------------- | ---- | ---- | ---- |
| BERT-TTA          | 0.05 | 0.15 | 0.80 |
| RNNContextual-TTA | 0.25 | 0.25 | 0.50 |
| AGLCF-TTA         | 0.05 | 0.35 | 0.60 |
| DHLG-TTA          | 0.05 | 0.35 | 0.60 |

