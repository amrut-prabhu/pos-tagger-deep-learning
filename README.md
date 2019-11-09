# POS Tagger using Deep Learning

## Running the program

**Training**:
```sh
python buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

# Example:
python buildtagger.py sents.train model-file
```

**Testing**:
```sh
python runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

# Example:
python runtagger.py sents.test model-file sents.out
```
python runtagger1.py pa2_blind_sents.test model-file1 pa2_blind_sents.out

**Evaluation**:
```sh
python eval.py <output_file_absolute_path> <reference_file_absolute_path>

# Example:
python eval.py sents.out sents.answer
```

## Penn Treebank Tagset

The list of POS tags can be seen [here](https://www.clips.uantwerpen.be/pages/mbsp-tags).

## Results

The following are the results when using `model-file` (Bidirectional LSTM model without character embeddings).

Parameters and Hyperparameters:
- LSTM
    - `dropout` = 0.5
    - `num_lstm_units` = 50 
    - `embedding_dim` = 32
- Batch size `b` = 32
- Max words in a sentence `max_length` = 250
- Num epochs = 7
- Learning rate `lr` of SGD optimizer
    - For epochs 0-3 = 0.1
    - For epochs 4-5 = 0.05
    - For epochs 6 = 0.005

Results of the POS tagger after training on `sents.train`.

| Test case    | Accuracy |
| ------------ | -------- |
| `sents.test` | 0.94266  |
| `2.train`    | 0.84498  |
| `2a.train`   | 0.82832  |
| `2b.train`   | 0.82223  |
| `3.train`    | 0.80777  |
| `4.train`    | 0.93860  |
| `5.train`    | 0.93513  |
