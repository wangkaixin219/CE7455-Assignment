from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_char_token = 0
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s ",
                "you are", "you re ", "we are", "we re ", "they are", "they re ")


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Char(object):
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: '*'}
        self.n_chars = 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        for char in word:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filter_pair(p): # e.g. p = ['go.', 'va !']
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def char_idx_from_sentence(char, sentence):
    indexes = [char_idx_from_word(char, word) for word in sentence.split(' ')]
    indexes.append([EOS_char_token])
    return indexes


def char_idx_from_word(char, word):
    return [char.char2index[character] for character in word]


def word_idx_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = word_idx_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair, input_lang, output_lang):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor


def read_languages(lang1, lang2, reverse=False):  # e.g. lang1 = eng, lang2 = fra
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
        input_char = Char(lang2)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        input_char = Char(lang1)

    return input_lang, output_lang, input_char, pairs


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, input_char, pairs = read_languages(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words and chars...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
        input_char.add_sentence(pair[0])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print("Counted chars:")
    print(input_char.name, input_char.n_chars)
    print('%d total pairs' % (len(pairs),))

    # Shuffle pairs and split into train_pairs and test_pairs
    random.shuffle(pairs)
    train_length = (len(pairs) // 10) * 8
    train_pairs = pairs[:train_length]
    test_pairs = pairs[train_length:]

    return input_lang, output_lang, input_char, train_pairs, test_pairs


if __name__ == '__main__':

    input_lang, output_lang, input_char, train_pairs, test_pairs = prepare_data('eng', 'fra', True)

    print('\nCounting total pairs, train pairs, and test pairs...')
    print('%d train pairs' % (len(train_pairs),))
    print('%d test pairs' % (len(test_pairs),))
    print('Example of a training pair: %s' % (train_pairs[0]))

    print('\nTesting charIndexesFromSentence...')
    sentence = 'je t aime'
    print('Sentence: %s' % sentence)
    print('Char Indexes: %s' % (char_idx_from_sentence(input_char, sentence)))

    print(input_char.char2index)
    print(input_char.char2count)
    print(input_char.index2char)
    print(len(input_char.char2index))
