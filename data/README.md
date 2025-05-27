## 📄 Data Preprocessing

To process the data, first refer to the examples in `process_data.py`, then run the script to generate the processed dataset.

### 📝 Processed Data Format

Each utterance in the processed dataset includes the following attributes:

| Attribute      | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `intent`       | The intent label of the utterance                                           |
| `text`         | The current utterance, tokenized into a list of words                      |
| `slots_labels` | Slot labels corresponding to each token in `text`                          |
| `speaker`      | Indicates the speaker of the utterance (e.g., user or system)              |
| `intent_shift` | Binary flag indicating whether the utterance shows an intent shift (1 = shift) |
| `index`        | The position of the utterance within the full dialogue                     |
| `start`        | Whether the utterance is the start of a dialogue (used only in MultiWOZ)   |

---

After running `process_data.py`, use the following command to compute contextual embeddings for all preceding utterances:

```bash
python get_tensor.py \
  --model_path "../pretrain_model/bert-base-uncased" \
  --input_file "sim/train.json" \
  --max_length 128
```

This will generate train.pkl and test.pkl files, which will be used in the training and evaluation process.