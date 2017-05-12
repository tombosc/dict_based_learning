#!/usr/bin/env python
from __future__ import print_function

import io
import numpy
import argparse

from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary

def main():
    parser = argparse.ArgumentParser("Analyze coverage of either a dictionary or pretrained embeddings on a given vocab.")
    parser.add_argument(
        "--dict", default="", help="Dictionary.")
    parser.add_argument(
        "--embedding", default="",
        help="Path to embeddings. Can either be a npy file or a raw glove txt file.")
    parser.add_argument(
        "--top_k", type=int, default=0,
        help="Optional, provide statistics for excluding top_k words from source (either dict or embedding)")
    parser.add_argument("--step_size", type=int, help="Report each", default=10000)
    parser.add_argument("--uncovered", help="Destination for uncovered files")
    parser.add_argument("vocab", help="Vocabulary")
    args = parser.parse_args()

    assert(args.vocab.endswith(".txt"))

    vocab = Vocabulary(args.vocab)
    words = vocab.words
    freqs = numpy.array(vocab.frequencies)
    total = float(freqs.sum())
    coverage = numpy.cumsum(freqs) / total
    print("Cumulative distribution:")
    for i in range(args.step_size, args.step_size * (len(freqs) / args.step_size), args.step_size):
        print(i, coverage[i] * 100)

    if not args.dict and not args.embedding:
        return

    uncovered_file = io.open('/dev/null', 'w')
    if args.uncovered:
        uncovered_file = io.open(args.uncovered, 'w', encoding='utf-8')

    if args.dict and args.top_k:
        print("Analysing coverage of dict of text")

    n_covered = 0
    n_covered_by_lowercasing = 0
    if args.dict:
        source_name = "dictionary"
        dict_ = Dictionary(args.dict)
        print("Dictionary has {} entries".format(dict_.num_entries()))

        n_more_def_than_1 = 0
        for i in range(args.top_k, len(freqs)):
            if len(dict_.get_definitions(words[i])) > 1:
                n_more_def_than_1 += freqs[i]
            if dict_.get_definitions(words[i]):
                n_covered += freqs[i]
            elif dict_.get_definitions(words[i].lower()):
                n_covered_by_lowercasing += freqs[i]
    elif args.embedding:
        source_name = "glove embeddings"
        # Loading (note: now only supports GloVe format)
        word_set = set([])
        if args.embedding.endswith(".txt"):
            with open(args.embedding) as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    word_set.add(word)
                f.close()
        elif args.embedding.endswith(".npy"):
            print("Warning: assuming that embeddings from .npy file are ordered according to the same vocabulary file as the one passed (using pack_glove --vocab vocab_passed_here)")
            emb_matrix = numpy.load(args.embedding)
            for i, emb in enumerate(emb_matrix):
                if not numpy.all(emb == 0):
                    word_set.add(words[i])

        print("Glove embeddings has {} entries".format(len(word_set)))

        for i in range(args.top_k, len(freqs)):
            if words[i] in word_set:
                n_covered += freqs[i]
            elif words[i].lower() in word_set:
                n_covered_by_lowercasing += freqs[i]
    else:
        raise NotImplementedError()


    print("Analysing coverage of " + source_name)
    if args.top_k:
        print("The first " + str(args.top_k) + " ranked words are covered by word embeddings.")
        print("This amounts to " + str(100*coverage[args.top_k - 1]) + "% of occurences.")
    else:
        print("Case when no word embeddings are used (args.top_k=0). " + source_name + " provides all embeddings")
    print(source_name + " covers {} % of total occurences".
          format(100 * n_covered / total))
    print("Querying not found words as lowercased words additionally covers {}% of total occurences".
          format(100 * n_covered_by_lowercasing / total))

    if args.top_k:
        n_not_covered_by_embs = total * (1 - coverage[args.top_k - 1])
        print(source_name + " covers additional {}% of occurences not covered by word embeddings".
          format(100 * n_covered / n_not_covered_by_embs))
        print("Querying not found words as lowercased words additionally covers {}% of occurences not covered by word embeddings".
          format(100 * n_covered_by_lowercasing / n_not_covered_by_embs))
        if args.dict:
            print("Occurences of dictionary defs with >1 def not covered by word embeddings: {}%".
              format(100 * n_more_def_than_1 / n_not_covered_by_embs))

    uncovered_file.close()

if __name__ == "__main__":
    main()
