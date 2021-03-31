#  Neural Machine Translation

This example trains a Seq2Seq model with attention. The dataset for this exercise is a machine translation dataset provided by Pytorch tutorial. 
The dataset includes 135,842 English-French sentence pairs. We read the text from the file line by line, and split a line into two parts by '\t', 
one of which is English sentence and the other is French one. 

## Train language model

The model will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

```bash 
python main.py                                          # Train a model with default hyperparameters
```

The scripts will train the model on training dataset, generate translations on test dataset automatically. Finally, it will calculate the BLEU scores. 
Since during each time, the training samples will randomly choose from the whole dataset, there will be a little difference between the outputs of the scores.
