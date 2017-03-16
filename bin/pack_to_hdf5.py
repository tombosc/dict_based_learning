#!/usr/bin/env python

import argparse

from dictlearn.h5py_conversion import text_to_h5py_dataset

def main():
    parser = argparse.ArgumentParser("Builds a dictionary")
    parser.add_argument("text", help="The text to use")
    parser.add_argument("h5", help="Destination")
    args = parser.parse_args()

    text_to_h5py_dataset(args.text, args.h5)

if __name__ == "__main__":
    main()
