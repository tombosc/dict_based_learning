#!/usr/bin/env python

import argparse
import logging
import numpy

from os import path
from dictlearn.vocab import Vocabulary

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Converts GLOVE embeddings to a numpy array")
    parser.add_argument("txt", help="GLOVE data in txt format")
    parser.add_argument("npy", help="Destination for npy format")
    parser.add_argument("--vocab", default="", help="Performs subsetting based on pased vocab")
    args = parser.parse_args()

    if args.vocab == "":
        embeddings = []
        dim = None
        with open(args.txt) as src:
            for i, line in enumerate(src):
                tokens = line.strip().split()
                features = map(float, tokens[1:])
                dim = len(features)
                embeddings.append(features)
                if i and i % 100000 == 0:
                    print i
        embeddings = [[0.] * dim] * len(Vocabulary.SPECIAL_TOKEN_MAP) + embeddings
        numpy.save(args.npy, embeddings)
    else:
        vocab = Vocabulary(args.vocab)
        print('Computing GloVe')

        # Loading
        embeddings_index = {}
        f = open(args.txt)
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # Embedding matrix
        embedding_matrix = numpy.zeros((vocab.size(), 300))
        for word in vocab._word_to_id:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[vocab.word_to_id(word)] = embedding_vector
            else:
                print('Missing from GloVe: {}'.format(word))

        numpy.save(args.npy, embedding_matrix)

if __name__ == "__main__":
    main()
