import re
import codecs
import os
import _pickle as cPickle
import numpy as np


class Cleaner(object):
    def __init__(self, args):
        self.train = os.path.join(args.data_dir, 'eng.train')
        self.dev = os.path.join(args.data_dir, 'eng.testa')
        self.test = os.path.join(args.data_dir, 'eng.testb')
        self.zeros = args.zeros
        self.tag_scheme = args.tag

    @staticmethod
    def zero_digits(s):
        return re.sub(r'\d', '0', s)

    def load_sentences(self, path):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
        """
        sentences = []
        sentence = []
        for line in codecs.open(path, 'r', 'utf8'):
            line = self.zero_digits(line.rstrip()) if self.zeros else line.rstrip()
            if not line:
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:
                        sentences.append(sentence)
                    sentence = []
            else:
                word = line.split()
                assert len(word) >= 2
                sentence.append(word)
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
        return sentences

    @staticmethod
    def is_bio(tags):
        """
        Check that tags have a valid BIO format.
        Tags in BIO1 format are converted to BIO2.
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            if len(split) != 2 or split[0] not in ['I', 'B']:
                return False
            if split[0] == 'B':
                continue
            elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
            elif tags[i - 1][1:] == tag[1:]:
                continue
            else:  # conversion IOB1 to IOB2
                tags[i] = 'B' + tag[1:]
        return True

    @staticmethod
    def bio2bioes(tags):
        """
        the function is used to convert
        BIO -> BIOES tagging
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'B':
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif tag.split('-')[0] == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        return new_tags

    def update_tag_scheme(self, sentences):
        """
        Check and update sentences tagging scheme to BIO2
        Only BIO1 and BIO2 schemes are accepted for input data.
        """
        for i, s in enumerate(sentences):
            tags = [w[-1] for w in s]
            # Check that tags are given in the BIO format
            if not self.is_bio(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in BIO format! Please check sentence %i:\n%s' % (i, s_str))
            if self.tag_scheme == 'BIOES':
                new_tags = self.bio2bioes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[-1] = new_tag
            else:
                raise Exception('Wrong tagging scheme!')

    def clean(self):
        train_s = self.load_sentences(self.train)
        test_s = self.load_sentences(self.test)
        dev_s = self.load_sentences(self.dev)

        self.update_tag_scheme(train_s)
        self.update_tag_scheme(test_s)
        self.update_tag_scheme(dev_s)

        return train_s, dev_s, test_s


class Dataset(object):
    def __init__(self, raw_train, raw_dev, raw_test, args):
        self.start = args.start
        self.stop = args.stop
        self.lower = args.lower
        self.mapping_pkl = os.path.join(args.data_dir, 'mapping.pkl')
        self.embedding_dim = args.embedding_dim
        self.embedding_path = os.path.join(args.embedding_dir, 'glove.6B.' + str(self.embedding_dim) + 'd.txt')

        dico_words, self.word_to_id, id_to_word = self.word_mapping(raw_train)
        dico_chars, self.char_to_id, id_to_char = self.char_mapping(raw_train)
        dico_tags, self.tag_to_id, id_to_tag = self.tag_mapping(raw_train)

        self.train_data = self.prepare_dataset(raw_train, self.word_to_id, self.char_to_id, self.tag_to_id)
        self.dev_data = self.prepare_dataset(raw_dev, self.word_to_id, self.char_to_id, self.tag_to_id)
        self.test_data = self.prepare_dataset(raw_test, self.word_to_id, self.char_to_id, self.tag_to_id)
        self.word_embedding = self.load_word_embedding()

    def lower_case(self, x):
        return x.lower() if self.lower else x

    @staticmethod
    def create_dico(item_list):
        """
        Create a dictionary of items from a list of list of items.
        """
        assert type(item_list) is list
        dico = {}
        for items in item_list:
            for item in items:
                if item not in dico:
                    dico[item] = 1
                else:
                    dico[item] += 1
        return dico

    @staticmethod
    def create_mapping(dico):
        """
        Create a mapping (item to ID / ID to item) from a dictionary.
        Items are ordered by decreasing frequency.
        """
        sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
        id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
        item_to_id = {v: k for k, v in id_to_item.items()}
        return item_to_id, id_to_item

    def word_mapping(self, sentences):
        """
        Create a dictionary and a mapping of words, sorted by frequency.
        """
        words = [[self.lower_case(x[0]) for x in s] for s in sentences]
        dico = self.create_dico(words)
        dico['<UNK>'] = 10000000  # UNK tag for unknown words
        word_to_id, id_to_word = self.create_mapping(dico)
        print("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in words)))
        return dico, word_to_id, id_to_word

    def char_mapping(self, sentences):
        """
        Create a dictionary and mapping of characters, sorted by frequency.
        """
        chars = ["".join([w[0] for w in s]) for s in sentences]
        dico = self.create_dico(chars)
        char_to_id, id_to_char = self.create_mapping(dico)
        print("Found %i unique characters" % len(dico))
        return dico, char_to_id, id_to_char

    def tag_mapping(self, sentences):
        """
        Create a dictionary and a mapping of tags, sorted by frequency.
        """
        tags = [[word[-1] for word in s] for s in sentences]
        dico = self.create_dico(tags)
        dico[self.start] = -1
        dico[self.stop] = -2
        tag_to_id, id_to_tag = self.create_mapping(dico)
        print("Found %i unique named entity tags" % len(dico))
        return dico, tag_to_id, id_to_tag

    def prepare_dataset(self, sentences, word_to_id, char_to_id, tag_to_id):
        """
        Prepare the dataset. Return a list of lists of dictionaries containing:
            - word indexes
            - word char indexes
            - tag indexes
        """
        data = []
        for s in sentences:
            str_words = [w[0] for w in s]
            words = [word_to_id[self.lower_case(w) if self.lower_case(w) in word_to_id else '<UNK>'] for w in str_words]
            # Skip characters that are not in the training set
            chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]
            tags = [tag_to_id[w[-1]] for w in s]
            data.append({
                'str_words': str_words,
                'words': words,
                'chars': chars,
                'tags': tags,
            })
        return data

    def load_word_embedding(self):
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(self.embedding_path, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == self.embedding_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.zeros(shape=(len(self.word_to_id), self.embedding_dim))

        for w in self.word_to_id:
            if w in all_word_embeds:
                word_embeds[self.word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[self.word_to_id[w]] = all_word_embeds[w.lower()]

        with open(self.mapping_pkl, 'wb') as f:
            mappings = {
                'word_to_id': self.word_to_id,
                'tag_to_id': self.tag_to_id,
                'char_to_id': self.char_to_id,
                'word_embeds': word_embeds,
            }
            cPickle.dump(mappings, f)

        print('word_to_id: ', len(self.word_to_id))
        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
        return word_embeds
