import os
import torch
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, dic):
        self.dictionary = Dictionary()
        if dic is not None:
            self.dictionary.word2idx = dic.word2idx
            self.dictionary.idx2word = dic.idx2word
        self.MAX_LENGTH = None
        self.train = self.tokenize(os.path.join(path, 'train.txt'), dic)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), dic)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), dic)

    def tokenize(self, path, dic):
        """Tokenize a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary,
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # append <eos> to every sample sequence
                words = line.split() + ['<eos>']
                # Calculate max sequence length
                if self.MAX_LENGTH is None:
                    self.MAX_LENGTH = len(
                        line[:line.find('<EOL>')+5].split(' '))
                else:
                    self.MAX_LENGTH = max(
                        len(line[:line.find('<EOL>') + 5].split(' ')),
                        self.MAX_LENGTH)
                tokens += len(words)
                # only add if not provided dictionary
                if dic is None:
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                # append <eos> to every sample sequence
                words = line.split() + ['<EOS>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
