#!/usr/local/bin/python

import numpy as np
from numpy.random import normal

def softmax(v, T):
    exp_v = np.exp(v/T)
    return exp_v / np.sum(exp_v)

class LogLinearModel():
    def __init__(self, vocabulary_size, features_size, markov_order, 
                 temperature=1.0):
        """
        markov_order: integer at least 1 such that
            p(x_t|x_t-1:x_1) = p(x_t|x_t-1:x_t-markov_order)
        temperature: temperature for softmax
        """
        assert(features_size >= 1)
        self.mo = markov_order
        self.V = vocabulary_size
        self.params = normal(0,1,(markov_order*features_size, vocabulary_size))
        self.features = normal(0,1,(vocabulary_size, features_size))
        self.initial_features = normal(0,1,(markov_order, features_size))
        self.T = temperature

    def proba(self, features):
        """
        return a categorical probability distribution over the vocabulary
        """
        product = np.dot(features, self.params)
        return softmax(product, self.T)
        
    def sample(self, n):
        """
        returns n sample tokens from the model
        """
        features = [f for f in self.initial_features]
        samples = []
        for i in range(1,n):
            f = np.concatenate(features[-self.mo:])
            s = np.random.choice(self.V, p=self.proba(f))
            samples.append(s)
            features.append(self.features[s])
        return samples

if __name__ == "__main__":
    model = LogLinearModel(100, 3, 2)
    print model.sample(10) 
