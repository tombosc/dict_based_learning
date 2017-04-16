#!/usr/bin/env python

import h5py
import argparse
import logging

from dictlearn.vocab import Vocabulary

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Builds a dictionary")
    parser.add_argument("--top_k", type=int, help="Top most frequent words to leave")
    parser.add_argument("text", help="The text to use")
    parser.add_argument("vocab", help="Destination")
    args = parser.parse_args()

    if args.text.endswith('.h5'):
        with h5py.File(args.text) as h5_file:
            text = h5_file['text'][:]
    else:
        text = args.text
    vocab = Vocabulary.build(text, args.top_k)
    vocab.save(args.vocab)

if __name__ == "__main__":
    main()
