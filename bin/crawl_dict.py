#!/usr/bin/env python

import argparse
import logging

from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary

def main():
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Crawl definitions for a vocabulary")
    parser.add_argument("--api_key",
                        help="Wordnik API key to use")
    parser.add_argument("--just-lemmas", action="store_true",
                        help="Just use the lemmas as the definition")
    parser.add_argument("--identity", action="store_true",
                        help="Identity mapping dictionary")
    parser.add_argument("--spelling", action="store_true",
                        help="Spelling dictionary")
    parser.add_argument("vocab", help="Vocabulary path")
    parser.add_argument("dict", help="Destination path for the dictionary")
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    dict_ = Dictionary(args.dict)
    if args.api_key:
        dict_.crawl_wordnik(vocab, args.api_key)
    elif args.just_lemmas:
        dict_.crawl_lemmas(vocab)
    elif args.identity:
        dict_.setup_identity_mapping(vocab)
    elif args.spelling:
        dict_.setup_spelling(vocab)
    else:
        raise ValueError("don't know what to do")

if __name__ == "__main__":
    main()
