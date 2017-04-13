#!/usr/bin/env python

import argparse
import logging

import numpy

from dictlearn.vocab import Vocabulary

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Converts GLOVE embeddings to a numpy array")
    parser.add_argument("txt", help="GLOVE data in txt format")
    parser.add_argument("npy", help="Destination for npy format")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
