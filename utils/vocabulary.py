import json

class Vocabulary:
    def __init__(self):
        self.word2idx = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_count = 4

    def add_sentence(self, sentence):
        '''Adds all words in a sentence to the vocabulary.'''
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.word_count
                self.idx2word[self.word_count] = word
                self.word_count += 1
    def sentence_to_indices(self, sentence):
        '''Converts a sentence to a list of indices based on the vocabulary.'''
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence.split()]
    
    def indices_to_sentence(self, indices):
        '''Converts a list of indices back to a sentence.'''
        return ' '.join([self.idx2word.get(idx, "<UNK>") for idx in indices])
    
    def __len__(self):
        return self.word_count