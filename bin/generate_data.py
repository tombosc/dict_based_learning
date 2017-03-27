#!/usr/bin/env python

import numpy as np
import os
import argparse
import json
import pickle

from dictlearn.generate_synthetic_data import FakeTextGenerator
from tests.util import temporary_content_path
from dictlearn.vocab import Vocabulary

def write_data(path, data):
    with open(path, "w") as f:
        f.write(data)

def main():
    parser = argparse.ArgumentParser("Generate synthetic data and outputs in files")
    parser.add_argument("path", type=str, help="Top most frequent words to leave")
    parser.add_argument("vocab_size", type=int, help="Vocabulary size")
    parser.add_argument("features_size", type=int, help="Features size")
    parser.add_argument("markov_order", type=int, help="Markov order")
    parser.add_argument("n_sentences", type=int, help="# sentences")
    parser.add_argument("pc_train", type=float, help="% train sentences")
    parser.add_argument("pc_valid", type=float, help="% valid sentences")
    parser.add_argument("pc_homonyms", type=float, default=0.2, help="% valid sentences")
    parser.add_argument("sample_temperature", type=float, default=1.0, help="% valid sentences")
    parser.add_argument("markov_order_dict", type=int, default=1)
    parser.add_argument("min_sentence_len", type=int, default=6)
    parser.add_argument("max_sentence_len", type=int, default=20)
    parser.add_argument("min_def_len", type=int, default=6)
    parser.add_argument("max_def_len", type=int, default=20)

    args = parser.parse_args()

    print "Number of sentences:", args.n_sentences
    assert(0 < args.pc_train + args.pc_valid < 1)
    assert(os.path.exists(args.path) == False)
    os.makedirs(args.path)
    args.pc_test = 1 - (args.pc_train + args.pc_valid)

    gen = FakeTextGenerator(args.vocab_size, args.features_size,
                            args.markov_order, args.sample_temperature,
                            args.pc_homonyms, args.markov_order_dict,
                            args.min_def_len, args.max_def_len)

    diff_len = args.max_sentence_len - args.min_sentence_len
    sentences_len = [np.random.choice(diff_len) + args.min_sentence_len for _ in range(args.n_sentences)]
    data = [' '.join(gen.sample_sentence(l)) for l in sentences_len]

    with temporary_content_path('\n'.join(data)) as path:
        vocab = Vocabulary.build(path)
        vocab.save(os.path.join(args.path, "vocab.txt"))

    dict_json = json.dumps(gen.dictionary, indent=4, sort_keys=True)
    write_data(os.path.join(args.path, "dict.json"), dict_json)

    train_data = data[:int(args.pc_train * args.n_sentences)]
    valid_data = data[len(train_data):len(train_data) + int(args.pc_valid * args.n_sentences)]
    test_data = data[-int(args.pc_test * args.n_sentences):]

    write_data(os.path.join(args.path, "train.txt"), '\n'.join(train_data))
    write_data(os.path.join(args.path, "valid.txt"), '\n'.join(valid_data))
    write_data(os.path.join(args.path, "test.txt"), '\n'.join(test_data))

    args_json = json.dumps(vars(args), indent=4, sort_keys=True)
    write_data(os.path.join(args.path, "params.json"), args_json)

    write_data(os.path.join(args.path, "generator.p"), pickle.dumps(gen))


if __name__ == "__main__":
    main()
