# Named Entity Recognition Language Modeling

Named Entity Recognition (NER) is an important information extraction task that requires to identify and classify named entities in a given text. 
These entity types are usually predefined like location, organization, person, and time. In this exercise, you will learn how to develop a neural NER model 
in Pytorch.

## Train language model

Before running the code, remember to download the word embedding file by the following command

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
```

The model will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

```bash 
# Train a model with LSTM character-level, 1-layer CNN word-level encoder, CRF decoder
python main.py --cuda --crf --char_encoder LSTM --word_encoder CNN      

# Train a model with CNN character-level, 2-layer CNN word-level encoder, CRF decoder
python main.py --cuda --crf --char_encoder CNN --word_encoder CNN2     

# Train a model with CNN character-level, dilated CNN word-level encoder, Softmax decoder
python main.py --cuda --char_encoder CNN --word_encoder CNN_DILATED     
```

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help                show this help message and exit
  --data_dir DIR            location of the data corpus
  --model_dir DIR           model directory
  --save_path PATH          path to the model
  --tag BIO/BIOES           tag scheme: BIO or BIOES
  --lower True/False        lower case or not
  --zeros True/False        zero digits or not
  --cuda                    use CUDA
  --start TAG               start tag, <START> default
  --stop TAG                stop tag, <STOP> default
  --seed SEED               random seed
  --embedding_dir DIR       embedding directory
  --embedding_dim DIM       embedding dimension
  --hidden_dim DIM          hidden dimension
  --dropout RATIO           dropout ratio
  --crf                     use CRF layer
  --pretrain True/False     use pretrained word embedding
  --char_encoder ENC        ENC = LSTM/CNN
  --word_encoder ENC        ENC = LSTM/CNN/CNN2/CNN3/CNN_DILATED
  --lr LR                   learning rate
  --epoch EPOCH             number of epochs
  ---clip CLIP              gradient clip
  --momentum RATIO          optimizer momentum
  --decay_rate RATIO        decay rate
  --plot_interval STEP      plot every # steps
  
```
