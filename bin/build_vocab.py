#!/usr/bin/env python

import argparse

from dictlearn.vocab import Vocabulary

def main():
  parser = argparse.ArgumentParser("Builds a dictionary")
  parser.add_argument("--top_k", type=int, help="Top most frequent words to leave")
  parser.add_argument("text", help="The text to use")
  parser.add_argument("vocab", help="Destination")
  args = parser.parse_args()

  vocab = Vocabulary.build(args.text, args.top_k)
  vocab.save(args.vocab)

if __name__ == "__main__":
    main()
