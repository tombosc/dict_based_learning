#!/usr/bin/env python

import argparse
import logging

from dictlearn.h5py_conversion import (
    text_to_h5py_dataset, squad_to_h5py_dataset, snli_to_h5py_dataset)
from dictlearn.corenlp import start_corenlp
from dictlearn.util import get_free_port

from blocks.bricks.cost import CategoricalCrossEntropy

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Converts text to HDF5")
    parser.add_argument("--type", choices=("text", "squad", "snli"), default='text',
                        help="What kind of data should be converted")
    parser.add_argument("data", help="The data to convert")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("h5", help="Destination")
    args = parser.parse_args()

    if args.type == 'text':
        if args.lowercase:
            raise NotImplementedError() # Just to be safe
        text_to_h5py_dataset(args.data, args.h5)
    elif args.type == 'squad':
        if args.lowercase:
            raise NotImplementedError() # Just to be safe
        port = get_free_port()
        try:
            corenlp = start_corenlp(port)
            squad_to_h5py_dataset(args.data, args.h5, "http://localhost:{}".format(port))
        finally:
            if corenlp and corenlp.returncode is None:
                corenlp.kill()
    elif args.type == 'snli':
        snli_to_h5py_dataset(args.data, args.h5, lowercase=args.lowercase)
    else:
        raise NotImplementedError(args.type)

if __name__ == "__main__":
    main()
