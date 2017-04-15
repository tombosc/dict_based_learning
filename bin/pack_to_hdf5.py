#!/usr/bin/env python

import argparse
import logging

from dictlearn.vocab import Vocabulary
from dictlearn.h5py_conversion import (
    text_to_h5py_dataset, squad_to_h5py_dataset, add_words_ids_to_squad)
from dictlearn.corenlp import start_corenlp
from dictlearn.util import get_free_port

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Converts text to HDF5")
    parser.add_argument("--type", choices=("text", "squad"), default='text',
                        help="What kind of data should be converted")
    parser.add_argument("--vocab",
                        help="Digitize date with the given vocabulary")
    parser.add_argument("data", help="The data to convert")
    parser.add_argument("h5", help="Destination")
    args = parser.parse_args()

    if args.type == 'text':
        text_to_h5py_dataset(args.data, args.h5)
    elif args.type == 'squad':
        vocab = None
        if args.vocab:
            vocab = Vocabulary(args.vocab)
        port = get_free_port()
        try:
            corenlp = start_corenlp(port)
            squad_to_h5py_dataset(args.data, args.h5, "http://localhost:{}".format(port))
            add_words_ids_to_squad(args.h5, vocab)
        finally:
            if corenlp and corenlp.returncode is None:
                corenlp.kill()


if __name__ == "__main__":
    main()
