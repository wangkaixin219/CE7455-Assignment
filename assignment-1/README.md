# Feed Forward Language Modeling

This example trains a 3-layer FNN on a language modeling task. The training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text, and we compare the cosine similarity
distribution with human made distributions, using the wordsim353 dataset, provided. 

## Train language model

The model will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the  current model will be evaluated against the test dataset.

```bash 
python main.py                                          # Train a model with default hyperparameters           
python main.py --cuda --tied --save model_tied.pt       # Train a tied model with CUDA and save the model as model_tied.pt 
python main.py --embed 300 --hidden 300 --lr 0.01       # Train a model with embedding and hidden dimension 300 using learning rate 0.01
python main.py --data other_dataset                     # Train a model with other dataset
```

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --embed EMBED         size of word embeddings
  --hidden HIDDEN       number of hidden units per layer
  --lr LR               initial learning rate
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --window_size W       sequence length
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save PATH           path to save the final model
```

## Generate texts
After you trained the model, you can generate your own text using `generate.py`.

```bash
python generate.py                                      # Generate text from the trained model with default hyperparameter
python generate.py --cuda --checkpoint ./model_tied.pt  # Generate text from the trained model with name model_tied.pt using CUDA
python generate.py --outfile myfile --words 2000        # Generate text with 2000 words and store it to myfile
```

The `generate.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data                location of the data corpus
  --checkpoint          model checkpoint to use
  --outfile             output file for generated text
  --words               number of words to generate
  --seed                random seed
  --cuda                use CUDA
  --temperature         temperature - higher will increase diversity
  --log-interval        reporting interval
```

## Spearman report
To evaluate the model you trained, you can use `spearman.py` to check. 

```bash
python spearman.py                                  # Evaluate the model with default hyperparameter
python spearman.py --cuda --model ./model_tied.pt   # Evaluate the model named model_tied.pt using CUDA
```

The `spearman.py` script accepts the following arguments:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --data                location of the similarity file
  --checkpoint          path to the model
  --cuda                use CUDA
```
