#!/usr/bin/env python
"""
Small utility for merging def and normal vocab

Run as:

python bin/merge_vocab.py --target_coverage_text=0.95 --target_coverage_def=0.9
--vocab_text=$DATA_DIR/snli/vocab.txt --vocab_def=$DATA_DIR/snli/dict_all_3_05_lowercase_lemma_vocab.txt
--target=$DATA_DIR/snli/dict_all_3_05_lowercase_lemma_vocab_0.95_0.9.txt
"""

import h5py
import argparse
import logging
from six import text_type
import json
import numpy as np

from dictlearn.vocab import Vocabulary


def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Builds a dictionary")
    parser.add_argument("--target_coverage_text", type=float, help="Target coverage of text")
    parser.add_argument("--target_coverage_def", type=float, help="Target coverage of def")
    parser.add_argument("--vocab_text", type=str, help="Vocabulary of text")
    parser.add_argument("--vocab_def", type=str, help="Vocabulary of def")
    parser.add_argument("--step_size", type=int, default=30)
    parser.add_argument("--target", type=str, default="Final path")
    args = parser.parse_args()

    vocab_text = Vocabulary(args.vocab_text)
    vocab_def = Vocabulary(args.vocab_def)

    # Greedy solution is optimal
    # I also approximate greedy a bit by adding word by word. This is fine, vocabs are big
    target_coverage_text = np.sum(vocab_text.frequencies) * args.target_coverage_text
    target_coverage_def = np.sum(vocab_def.frequencies) * args.target_coverage_def
    current_vocab = set([])


    # Of course I could use binsearch
    for id in range(vocab_def.size() / args.step_size):
        for id2 in range(args.step_size):
            current_vocab.add(vocab_def.id_to_word(id*args.step_size + id2))

        current_vocab_mod = set(current_vocab)

        current_coverage_def = 0.0
        current_coverage_text = 0.0

        for w in current_vocab_mod:
            current_coverage_def += vocab_def.frequencies[vocab_def.word_to_id(w)]
            current_coverage_text += vocab_text.frequencies[vocab_text.word_to_id(w)]

        id_text = 0
        while current_coverage_text < target_coverage_text:
            while vocab_text.id_to_word(id_text) in current_vocab_mod:
                id_text += 1
                if id_text >= vocab_text.size():
                    raise Exception("Perhaps try lower target coverage")

            w = vocab_text.id_to_word(id_text)
            current_vocab_mod.add(w)
            current_coverage_def += vocab_def.frequencies[vocab_def.word_to_id(w)]
            current_coverage_text += vocab_text.frequencies[id_text]

        if current_coverage_def > target_coverage_def:
            current_vocab = current_vocab_mod
            break

        print("After adding {} words I covered {} of def and {} of text occurences".format(
            len(current_vocab_mod), current_coverage_def / float(np.sum(vocab_def.frequencies))
            , current_coverage_text / float(np.sum(vocab_text.frequencies))
        ))

    # To be safe rechecking shortlist works
    current_coverage_def = 0
    current_coverage_text = 0
    for w in current_vocab:
        current_coverage_def += vocab_def.frequencies[vocab_def.word_to_id(w)]
        current_coverage_text += vocab_text.frequencies[vocab_text.word_to_id(w)]

    print("Sanity check: after adding {} words I covered {} of def and {} of text occurences".format(
        len(current_vocab), current_coverage_def / float(np.sum(vocab_def.frequencies))
        , current_coverage_text / float(np.sum(vocab_text.frequencies))
    ))

    vocab_result = Vocabulary.build(list(current_vocab), sort_by='lexicographical')
    vocab_result.save(args.target)

    # while current_coverage_def < target_coverage_def or \
        # current_coverage_text < target_coverage_text:
        #
        # if current_coverage_def < target_coverage_def:
        #     # Rewind until id_def is unique
        #     while vocab_def.id_to_word(id_def) in current_vocab:
        #         id_def += 1
        #         if id_def >= vocab_def.size():
        #             raise Exception("Perhaps try lower target coverage")
        #
        #     w = vocab_def.id_to_word(id_def)
        #     current_vocab.add(w)
        #     current_coverage_def += vocab_def.frequencies[id_def]
        #     current_coverage_text += vocab_text.frequencies[vocab_text.word_to_id(w)]
        #
        # if current_coverage_text < target_coverage_text:
        #     # Rewind until id_text is unique
        #     while vocab_text.id_to_word(id_text) in current_vocab:
        #         id_text += 1
        #         if id_text >= vocab_text.size():
        #             raise Exception("Perhaps try lower target coverage")
        #
        #     w = vocab_text.id_to_word(id_text)
        #     current_vocab.add(w)
        #     current_coverage_def += vocab_def.frequencies[vocab_def.word_to_id(w)]
        #     current_coverage_text += vocab_text.frequencies[id_text]
        #
        # print("After adding {} words I covered {} of def and {} of text occurences".format(
        #     len(current_vocab), current_coverage_def / float(np.sum(vocab_def.frequencies))
        #     , current_coverage_text / float(np.sum(vocab_text.frequencies))
        # ))

    # TODO: Sanity check that vocab indeed satisifes the printed out


if __name__ == "__main__":
    main()

