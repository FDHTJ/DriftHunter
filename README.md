# Intent Drift Detection in Continuous Conversation via Temporal Transition Accumulating(TTA)
📂 Dataset and Preprocessing Instructions

We use the following publicly available datasets in our experiments:

[ATIS(Airline Travel Information System)](https://github.com/howl-anderson/ATIS_dataset/)

[SIM(The Simulated-Dialogue dataset)](https://github.com/google-research-datasets/simulated-dialogue)

[Multiwoz(Multi-Domain Wizard-of-Oz Datase)](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)

Due to the large size of the processed MultiWOZ dataset, we only include the preprocessed versions of ATIS and SIM in our repository. However, we provide the necessary scripts to preprocess all datasets, including MultiWOZ.

The preprocessing script is located at:

```
/data/process_data.py
```
📌 Before running the script, we recommend merging all dialogues from the original datasets into a single file for each dataset.

Detailed preprocessing instructions can be found in:
```
/data/README.md
```
Once the data is prepared, you can run the training and evaluation pipeline using:

```
/train/train_and_eval_with_tta.py
```

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

