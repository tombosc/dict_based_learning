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
    parser.add_argument("text", help="The text to use. Pass file names separated by comma to concatenate texts")
    parser.add_argument("vocab", help="Destination")
    args = parser.parse_args()

    text = []
    for f_name in args.text.split(","):
        logging.info("Processing " + f_name)
        if f_name.endswith('.h5'):
            with h5py.File(f_name) as h5_file:
                text.extend(h5_file['text'][:])
        else:
            with open(f_name) as file_:
                def data():
                    for line in file_:
                        for word in line.strip().split():
                            yield word
                text.extend(data())
        logging.info("{} words".format(len(text)))

    vocab = Vocabulary.build(text, args.top_k)
    vocab.save(args.vocab)

if __name__ == "__main__":
    main()

