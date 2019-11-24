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
python runtagger.py 6.test model-file 6.out

**Evaluation**:
```sh
python eval.py <output_file_absolute_path> <reference_file_absolute_path>

# Example:
python eval.py 6.out 6.answer
```

## Penn Treebank Tagset

The list of POS tags can be seen [here](https://www.clips.uantwerpen.be/pages/mbsp-tags).

## Results

The following are the results when using `model-file` (Bidirectional LSTM model without character embeddings).
Training time is restricted to 10 minutes. 
Testing time for `6.test` is restricted to 2 minutes.

### Parameters and Hyperparameters 1:
- LSTM
    - `dropout` = 0.5
    - `num_lstm_units` = 50 
    - `embedding_dim` = 32
- Batch size `b` = 32
- Max words in a sentence `max_length` = 150
- Num epochs = 7
- Learning rate `lr` of SGD optimizer
    - For epochs 0-3 = 0.1
    - For epochs 4-5 = 0.05
    - For epoch 6 = 0.005

Results of the POS tagger after training on `sents.train`.

Final loss printed = 0.1626

| Test case    | Accuracy |
| ------------ | -------- |
| `sents.test` | 0.94363  |
| `2.test`     | 0.84554  |
| `2a.test`    | 0.83023  |
| `2b.test`    | 0.82338  |
| `3.test`     | 0.80642  |
| `4.test`     | 0.93334  |
| `5.test`     | 0.93843  |
| `6.test`     | 0.93548  |


### Parameters and Hyperparameters 2:
- LSTM
    - `dropout` = 0.5
    - `num_lstm_units` = 50 
    - `embedding_dim` = 32
- Batch size `b` = 32
- Max words in a sentence `max_length` = 150
- Num epochs = 7
- Learning rate `lr` of SGD optimizer
    - For epochs 0-3 = **0.25**
    - For epochs 4-5 = 0.05
    - For epoch 6 = 0.005

Results of the POS tagger after training on `sents.train`.

Final loss printed = 0.1316

| Test case    | Accuracy |
| ------------ | -------- |
| `sents.test` | 0.95234  |
| `2.test`     | 0.85604  |
| `2a.test`    | 0.84196  |
| `2b.test`    | 0.83723  |
| `3.test`     | 0.82133  |
| `4.test`     | 0.93972  |
| `5.test`     | 0.94421  |
| `6.test`     | 0.94344  |
