from __future__ import division

import numpy as np
import itertools
from numpy.random import normal

def softmax(v, T):
    exp_v = np.exp(v/T)
    return exp_v / np.sum(exp_v)

# TODO: consider refactoring the code? generating the dictionary is very similar 
# to generating the corpus with a few exceptions:
# - definitions are deterministic. Maybe sample w/ low temperature instead
# - the definitions are not conditionned on the word we are defining (whereas
#   sentences are not conditioned on anything). 

class FakeTextGenerator(object):
    """
    FakeTextModel is used to produce fake data that looks like text. It 
    consists in 2 language models:
    - one generates the corpus per se
    - one generates the definitions of the tokens used in the corpus

    token: string that represent a word as it will appear in the data

    feature: vector that is used by the model to predict next tokens
             the m last features where m is the markov order are sufficient 
             statistics. True data-generating embeddings.

    vocabulary: list of all the tokens
                CAREFUL: there can be several time the same tokens (homonyms)

    self.features and self.vocabulary are indexed by the same indices

    internally, we prefer the feature representation than the token repr
    because 2 tokens can have different meanings and thus embeddings (homonyms)
    """
    def __init__(self, vocabulary_size, features_size, markov_order, 
                 temperature=1.0, pc_double_meaning=0.2):
        """
        markov_order: integer at least 1 such that
            p(x_t|x_t-1:x_1) = p(x_t|x_t-1:x_t-markov_order)
        temperature: temperature for softmax
        pc_double_meaning: percent of the tokens that have 2 different possible
                           feature vectors, i.e. 2 meanings
        """
        assert(features_size >= 1)
        self.mo = markov_order
        self.V = vocabulary_size
        self.T = temperature

        self.mo_d = 7 # markov order for the dictionary definitions
        self.len_definition = 4
        self.params = normal(0,1,(self.mo * features_size, self.V))
        self.params_d = normal(0,1,(self.mo_d * features_size, self.V))
        # trying to regulate the norm of the features.
        self.features = normal(0,1/(2*np.log(features_size)),
                              (self.V, features_size))

        self.features_size = features_size

        # tokens are composed of a..z letters 
        alphabet = ''.join([chr(c) for c in range(97, 97+26)]) # str(a..z)
        # tokens all have the same size tok_len
        self.tok_len = int(np.log(vocabulary_size) / np.log(len(alphabet)) + 1)
        n_homonyms = int(self.V * pc_double_meaning)
        # enumerate all the tokens
        self.vocabulary = []
        for i, tok in zip(range(self.V - n_homonyms),
                          itertools.product(alphabet, repeat=self.tok_len)):
            self.vocabulary.append(''.join(tok))
        for i in range(n_homonyms):
            tok = np.random.choice(self.V - n_homonyms)
            self.vocabulary.append(self.vocabulary[tok])

        # create definitions and fill the dictionary
        self.dictionary = {}
        for i in range(self.V - n_homonyms):
            tok = self.vocabulary[i]
            self.dictionary[tok] = [self.define_token(i)]
        for i in range(self.V - n_homonyms, self.V):
            tok = self.vocabulary[i]
            self.dictionary[tok].append(self.define_token(i))

        self.initial_features = normal(0,1,(self.mo, features_size))
        self.initial_features_d = normal(0,1,(self.mo_d, features_size))

    def proba(self, features, params):
        """
        return a categorical probability distribution over the vocabulary
        """
        product = np.dot(features, params)
        return softmax(product, self.T)

    def define_token(self, token_idx):
        """
        define a token using its index
        """
        tokens = [self.vocabulary[token_idx]]
        features = [self.features[token_idx]]
        for i in range(self.len_definition):
            order = min(self.mo_d, len(tokens))
            feature = np.concatenate(features[-order:])
            params = self.params_d[-order*self.features_size:]
            proba = self.proba(feature, params)
            s = np.argmax(proba)
            # TODO: here we took the argmax. Maybe we should sample with low 
            # temperature?
            tokens.append(self.vocabulary[s])
            features.append(self.features[s])
        return tokens

    def sample_sentence(self, n):
        """
        returns a sample sentence of n tokens
        """
        tokens = []
        features = [f for f in self.initial_features[-1:]]
        for i in range(n):
            order = min(self.mo, len(features))
            feature = np.concatenate(features[-order:])
            params = self.params[-order*self.features_size:]
            proba = self.proba(feature, params)
            s = np.random.choice(self.V, p=proba)
            tokens.append(self.vocabulary[s])
            features.append(self.features[s])
        return tokens

if __name__ == "__main__":
    model = FakeTextGenerator(100, 3, 2)
    print "sentences:"
    for i in range(10):
        print model.sample_sentence(10) 
    print "definitions:"
    for token, definition in model.dictionary.iteritems():
        print "token:", token, ":", definition
