from __future__ import division

import numpy as np
import itertools
from numpy.random import normal, uniform
from collections import defaultdict, Counter
from random import shuffle
from util import softmax

class FakeTextGenerator(object):
    """
    FakeTextModel is used to produce fake data that looks like text. 

    This is a really simple version where:
    - there are no homonyms.
    - the vocabulary is split in 2 between primes and composed.
      primes
      composed: their definition is a sequence of prime tokens
                their feature vectors is the mean of the features 
                of the tokens in their definitions (possibly rescaled)

    The model is a mixture model that chooses either from primes or composed tokens.

    token: string that represent a word as it will appear in the data

    feature: vector that is used by the model to predict next tokens
             the m last features where m is the markov order are sufficient 
             statistics. True data-generating embeddings.

    vocabulary: list of all the tokens

    self.features and self.vocabulary are indexed by the same indices

    internally, we prefer the feature representation than the token repr
    because 2 tokens can have different meanings and thus embeddings (homonyms)
    """
    def __init__(self, n_primes, n_composed, features_size, markov_order, 
                 temperature=1.0, min_len_definitions=2, max_len_definitions=4):
        """
        markov_order: integer at least 1 such that
            p(x_t|x_t-1:x_1) = p(x_t|x_t-1:x_t-markov_order)
        temperature: temperature for softmax
        """
        self.mo = markov_order
        self.np = n_primes
        self.nc = n_composed
        self.V = self.np + self.nc
        self.T = temperature
        self.min_len_def = min_len_definitions
        self.max_len_def = max_len_definitions
        self.features_size = features_size

        # tokens are composed of a..z letters 
        alphabet = ''.join([chr(c) for c in range(97, 97+26)]) # str(a..z)
        # tokens all have the same size tok_len
        self.tok_len = int(np.log(self.V) / np.log(len(alphabet)) + 1)
        # enumerate all the tokens
        self.vocabulary = []
        for i, tok in zip(range(self.V),
                          itertools.product(alphabet, repeat=self.tok_len)):
            self.vocabulary.append(''.join(tok))

        self.params = uniform(0,1,(self.mo * features_size, self.V))
        self.features = uniform(0,1,(self.V,features_size))
        self.dictionary = {}
        for i in range(self.np, self.np+self.nc):
            # sample len of def, sample def, store in dictionary
            # then compute the features as a rescaled mean of the features
            len_diff = self.max_len_def - self.min_len_def
            len_def = np.random.choice(len_diff) + self.min_len_def
            definition = np.random.choice(self.np, size=len_def, replace=False)
            tok = self.vocabulary[i]
            self.dictionary[tok] = [self.vocabulary[e] for e in definition]
            #factor = np.random.beta(a=3, b=2.5) # closer to 1 than 0
            #factor = np.random.beta(a=1, b=3) # closer to 0 than 1
            factor = 1#1/(8*self.nc)
            f = factor * np.mean([self.features[e] for e in definition], axis=0)
            self.features[i] = f

        self.initial_features = uniform(0,1,(self.mo, features_size))

    def proba(self, features, params):
        """
        return a categorical probability distribution over the vocabulary
        """
        product = np.dot(features, params)
        return softmax(product, self.T)

    def sample_sentence(self, max_num_np, avg_len, min_len, max_len):
        """
        returns a tuple (sentence, list of positions of non primes)
        """
        tokens = []
        features = [f for f in self.initial_features[-1:]]

        non_prime_pos = [] # store positions of non primes
        max_tokens = 100
        for i in range(max_tokens):
            order = min(self.mo, len(features))
            feature = np.concatenate(features[-order:])
            # mixture model
            p = 1-(1/avg_len)
            prime = np.random.choice(2, p=[1-p, p]) 
            if prime == 1:
                begin = 0
                end = self.np
            else:
                begin = self.np
                end = self.V
            params = self.params[-order*self.features_size:,begin:end]
            proba = self.proba(feature, params)
            s = np.random.choice(end-begin, p=proba) + begin
            if s > self.np: # s is a non prime
                non_prime_pos.append(i)
            if len(non_prime_pos) > max_num_np:
                break
            tokens.append(self.vocabulary[s])
            features.append(self.features[s])
        if min_len < len(tokens) < max_len:
            return tokens, non_prime_pos[:-1]
        else:
            return self.sample_sentence(max_num_np, avg_len, min_len, max_len)

    def create_corpus(self, n_sentences, min_len, max_len, train_ratio, 
                      valid_ratio):
        """
        Create a corpus (train, validation, test) using the model.

        max_len is unused here. TODO
        """
        sentences = []
        counts = Counter()
        contains = defaultdict(list)
        train, valid, test = set(), set(), set()
        n_words = 0
        counts_len = Counter()
        for i in range(int(n_sentences)):
            s, non_prime_pos = self.sample_sentence(1, 6, min_len, max_len)
            for p in non_prime_pos:
                counts[s[p]] += 1
                contains[s[p]].append(i)
            n_words += len(s)
            counts_len[len(s)] += 1
            sentences.append(s)
            #print "non primes", n_p, "over", len(s) 
        print counts_len
        print counts
        ordered_counts = counts.most_common()
        # good thing to leave this verbose probably...
        print "and stratify sample the sentences that contain them"
        print "the relative frequencies of these tokens is between",
        print ordered_counts[0][1]/n_words, "and", ordered_counts[-1][1]/n_words
        print "uniform probability would be:", 1/self.V
        shuffle(ordered_counts)
        b_train = int(len(ordered_counts)*train_ratio)
        b_valid = int(len(ordered_counts)*valid_ratio)
        print "b_train, b_valid", int(len(ordered_counts)), b_train, b_valid

        # we go through all the rare words
        # we put each sentence that contains this word into the train, test and
        # validation set.
        # although, we don't want that that words be in several sets because 
        # we want to observe generalisation on unseen words during training
        # so if a sentence is both in train and test (because it contains 
        # several rare words which belong to different sets) then we throw it
        # away.
        i = 0
        for w, _ in ordered_counts:
            for s in contains[w]:
                if i <= b_train:
                    train.add(s)
                elif i <= b_train + b_valid:
                    valid.add(s)
                else:
                    test.add(s)
            i += 1
        print "total of sentences", len(train), len(valid), len(test)

        get_sentences = lambda S: [sentences[i] for i in S]
        return (get_sentences(train), get_sentences(valid), get_sentences(test))

        
if __name__ == "__main__":
    model = FakeTextGenerator(1000, 20, 2)
    sentences = model.create_corpus(5000, 5, 10, 0.5, 0.3, 0.5)
