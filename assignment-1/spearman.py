import pandas as pd
import numpy as np
from numpy.linalg import norm
import torch
import data
import argparse
import scipy.stats as s

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Language Model - Word-Similarity Test')
    parser.add_argument('--data', type=str, default='./data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt', help='location of the similarity file')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--checkpoint', type=str, default='model.pt', help='path to the model')
    args = parser.parse_args()

    # =============== Load device ===============
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    # =============== Load model ===============
    with open(args.model, 'rb') as f:
        model = torch.load(f)
        model = model.to(device)

    corpus = data.Corpus(args.data)

    # =============== Load word data ===============
    names = ['Word 1', 'Word 2', 'Human (mean)']
    df = pd.read_table('data/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt', sep='\t', header=None, names=names)
    word_1 = df['Word 1']
    word_2 = df['Word 2']
    df['Indices 1'] = word_1.apply(lambda x: corpus.dictionary.word2idx[x] if x in corpus.dictionary.word2idx else None)
    df['Indices 2'] = word_2.apply(lambda x: corpus.dictionary.word2idx[x] if x in corpus.dictionary.word2idx else None)
    df = df.dropna().reset_index(drop=True)

    df['Indices 1'] = df['Indices 1'].astype('int')
    df['Indices 2'] = df['Indices 2'].astype('int')

    # =============== Get embedding ===============
    embedding_1 = df['Indices 1'].apply(lambda x: model.encoder.weight[x].detach().numpy())
    embedding_2 = df['Indices 2'].apply(lambda x: model.encoder.weight[x].detach().numpy())

    # =============== Compute cosine similarity ===============
    cosine_similarity = []
    for t1, t2 in zip(*(embedding_1, embedding_2)):
        cos_sim = t1.dot(t2) / (norm(t1) * norm(t2))
        cosine_similarity.append(cos_sim)
    df['Cosine Similarity'] = pd.Series(cosine_similarity)

    # Sort by Human (mean) and assign rank values
    df = df.sort_values(['Human (mean)']).reset_index(drop=True)
    df['Rank 1'] = np.arange(len(df)) + 1

    # Sort by Cosine Similarity
    df = df.sort_values(['Cosine Similarity']).reset_index(drop=True)
    df['Rank 2'] = np.arange(len(df)) + 1

    # Compute d and d^2 where d = difference between ranks and d^2 = difference squared
    df['d'] = df['Rank 1'] - df['Rank 2']
    df['d^2'] = df['d'] ** 2

    # Reindex dataframe columns
    df = df.reindex(columns=['Word 1', 'Word 2', 'Human (mean)', 'Cosine Similarity', 'Rank 1', 'Rank 2', 'd', 'd^2'])

    # =============== Compute Spearman correlation value ===============
    n = len(df)
    corr = 1 - (6 * df['d^2'].sum()) / (n * (n**2 - 1))
    print('The spearman correlation value is', corr)

    # check the correlation value
    spearman_corr, spearman_p  = s.spearmanr(df['Rank 1'], df['Rank 2'])
    print('The spearman correlation value is', spearman_corr)


