#!/usr/bin/env python

import argparse
import logging
import numpy

from os import path
from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Converts GLOVE embeddings to a numpy array")
    parser.add_argument("txt", help="GLOVE data in txt format")
    parser.add_argument("npy", help="Destination for npy format")
    parser.add_argument("--vocab", default="", help="Performs subsetting based on passed vocab")
    parser.add_argument("--dict", default="", help="Performs subsetting based on passed dict")

    # OOV handling
    parser.add_argument("--try-lemma", action="store_true", help="Try lemma")
    parser.add_argument("--try-lowercase", default="", help="Try lowercase")

    args = parser.parse_args()

    if args.dict and not args.vocab: 
        # usually you'd want to use both, I suppose
        raise NotImplementedError("Not implemented")
    if args.try_lemma or args.try_lowercase:
        # TODO(kudkudak): Implement
        raise NotImplementedError("Not implemented yet")

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
        if args.dict:
            dict_ = Dictionary(args.dict)
            
        print('Computing GloVe')

        # Loading
        embeddings_index = {}
        f = open(args.txt)
        print('Reading GloVe file')
        for line in f:
            values = line.split(' ')
            word = values[0]
            dim = len(values[1:])
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # Embedding matrix
        print('Reading GloVe file')
        embedding_matrix = numpy.zeros((vocab.size(), dim))
        for word in vocab._word_to_id:
            embedding_vector = embeddings_index.get(word)
            in_glove = embedding_vector is not None
            if args.dict:
                in_dict = len(dict_.get_definitions(word)) > 0

            if in_glove and (not args.dict or in_dict):
                # words not found in embedding index will be all-zeros.
                embedding_matrix[vocab.word_to_id(word)] = embedding_vector
            else:
                if not in_glove:
                    print(u'Missing from GloVe: {}'.format(word))
                else:
                    print(u'Missing from dict: {}'.format(word))


        numpy.save(args.npy, embedding_matrix)

if __name__ == "__main__":
    main()
