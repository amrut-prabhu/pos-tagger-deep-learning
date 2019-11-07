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

**Evaluation**:
```sh
python eval.py <output_file_absolute_path> <reference_file_absolute_path>

# Example:
python eval.py sents.out sents.answer
```

## Penn Treebank Tagset

The list of POS tags can be seen [here](https://www.clips.uantwerpen.be/pages/mbsp-tags).
