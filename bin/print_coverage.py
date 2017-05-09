#!/usr/bin/env python

import numpy
import argparse

from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary

def main():
    parser = argparse.ArgumentParser("Analyize text coverage")
    parser.add_argument("--dict", help="Dictionary", default="")
    parser.add_argument("--embedding", help="Path to embedding", default="")
    parser.add_argument("--top_k", type=int, help="Top k words from vocabulary")
    parser.add_argument("--try_lowercase", type=bool)
    parser.add_argument("--step_size", type=int, help="Report each", default=10000)
    parser.add_argument("vocab", help="Vocabulary")
    args = parser.parse_args()

    vocab = Vocabulary(args.vocab)
    words = vocab.words
    freqs = numpy.array(vocab.frequencies)
    total = float(freqs.sum())
    coverage = numpy.cumsum(freqs) / total
    for i in range(args.step_size, args.step_size * (len(freqs) / args.step_size), args.step_size):
        print(i, coverage[i] * 100)

    if args.dict and args.top_k:
        print("Analysing coverage of dict of text")
        dict_ = Dictionary(args.dict)
        n_not_covered_by_embs = total * (1 - coverage[args.top_k - 1])
        n_covered_by_dict = 0
        n_more_def_than_1 = 0
        n_covered_by_dict_by_lowercasing = 0
        n_covered_by_dict_by_lemmatizing = 0
        for i in range(args.top_k, len(freqs)):

            if len(dict_.get_definitions(words[i])) > 1:
                n_more_def_than_1 += freqs[i]

            if dict_.get_definitions(words[i]):
                n_covered_by_dict += freqs[i]
            elif dict_.get_definitions(words[i].lower()):
                n_covered_by_dict_by_lowercasing += freqs[i]
            else:
                pass


        print("Dictionary has {} entries".format(dict_.num_entries()))
        print("Dictionary covers {}% of total occurences in addition to word emb".
              format(100 * n_covered_by_dict / total))
        print("Dictionary covers additional {}% of total occurences not covered by word emb".
              format(100 * n_covered_by_dict / n_not_covered_by_embs))
        print("Dictionary def with >1 def (calculated only after topk) {}%".
              format(100 * n_more_def_than_1 / n_not_covered_by_embs))
        print("Querying dict with lowercased words covers {}% in addition to word emb".
              format(100 * n_covered_by_dict_by_lowercasing / total))
        print("Querying dict with lowercased words covers {}% of total occurences not covered by word emb".
              format(100 * n_covered_by_dict_by_lowercasing / n_not_covered_by_embs))

    elif args.embedding and args.top_k:

        print("Analysing coverage of emb")
        # Loading (note: now only supports GloVe format)
        word_set = set([])
        with open(args.embedding) as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                word_set.add(word)
            f.close()

        n_covered_by_dict = 0
        for i in range(args.top_k, len(freqs)):
            if words[i] in word_set or (args.try_lowercase and words[i].lower() in word_set):
                n_covered_by_dict += freqs[i]

        assert args.top_k is not None

        print("Embedding has {} entries".format(len(word_set)))
        print("Embedding covers fraction {} of total occurences".format(n_covered_by_dict / total))
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
