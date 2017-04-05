#!/usr/bin/env python

import numpy
import argparse

from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary

def main():
    parser = argparse.ArgumentParser("Analyize text coverage")
    parser.add_argument("--dict", help="Dictionary")
    parser.add_argument("--top_k", type=int, help="Top k words from vocabulary")
    parser.add_argument("vocab", help="Vocabulary")
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    words = vocab.words
    freqs = numpy.array(vocab.frequencies)
    total = float(freqs.sum())
    coverage = numpy.cumsum(freqs) / total
    for i in range(10000, 10000 * (len(freqs) / 10000), 10000):
        print(i, coverage[i] * 100)

    if args.dict and args.top_k:
        dict_ = Dictionary(args.dict)
        n_covered_by_dict = 0
        for i in range(args.top_k, len(freqs)):
            if dict_.get_definitions(words[i]):
                n_covered_by_dict += freqs[i]
        print("Dictionary has {} entries".format(dict_.num_entries()))
        print("Dictionary covers {}% in addition to the vocab".format(n_covered_by_dict / total))



if __name__ == "__main__":
    main()
