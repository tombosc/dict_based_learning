#!/usr/bin/env python
"""
Builds vocab from either h5/txt or dict.

Call as:
python bin/build_vocab.py $DATA_DIR/snli/dict_all_3_05_lowercase_lemma.json
    $DATA_DIR/snli/dict_all_3_05_lowercase_lemma_vocab.txt
"""

import h5py
import argparse
import logging
from six import text_type
import json
import collections

from dictlearn.vocab import Vocabulary


def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Builds a dictionary")
    parser.add_argument("--top-k", type=int, help="Top most frequent words to leave")
    parser.add_argument(
        "--weight-dict-entries", default=None,
        help="Weight dict entries according to the freqs from a vocab.")
    parser.add_argument(
        "--exclude-top-k", type=int,
        help="Ignore definitions of a number of most frequent words")
    parser.add_argument("text", help="The text to use. Pass file names separated by comma to concatenate texts")
    parser.add_argument("vocab", help="Destination")
    args = parser.parse_args()

    text = []
    if args.weight_dict_entries:
        text = collections.defaultdict(int)
        vocab_text = Vocabulary(args.weight_dict_entries)
    for f_name in args.text.split(","):
        logging.info("Processing " + f_name)
        if f_name.endswith('.h5'):
            with h5py.File(f_name) as h5_file:
                if 'text' not in h5_file.keys():
                    print("Missing text field from " + f_name)
                text.extend(h5_file['text'][:])
        elif f_name.endswith('.json'):
            logging.info("Will build the vocabulary from definitions in a dictionary")
            dict_ = json.load(open(f_name, "r"))
            for word, list_defs in dict_.items():
                if args.exclude_top_k and vocab_text.word_to_id(word) < args.exclude_top_k:
                    continue
                for def_ in list_defs:
                    if args.weight_dict_entries:
                        for def_word in def_:
                            text[def_word] += vocab_text.word_freq(word)
        else:
            with open(f_name) as file_:
                def data():
                    for line in file_:
                        for word in line.strip().split():
                            try:
                                yield text_type(word, 'utf-8')
                            except:
                                print("Skipped word " + word)
                text.extend(data())
        logging.info("{} words".format(len(text)))

    vocab = Vocabulary.build(text, args.top_k)
    vocab.save(args.vocab)

if __name__ == "__main__":
    main()

