#!/usr/bin/env python

import argparse
import logging

from dictlearn.vocab import Vocabulary
from dictlearn.h5py_conversion import add_words_ids_to_squad

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Digitizes text and add a vocab")
    parser.add_argument("vocab", help="Vocabulary")
    parser.add_argument("h5", help="Destination")
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    add_words_ids_to_squad(args.h5, vocab)


if __name__ == "__main__":
    main()
