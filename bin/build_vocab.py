#!/usr/bin/env python

import argparse

from dictlearn.vocab import Vocabulary

def main():
  parser = argparse.ArgumentParser("Builds a dictionary")
  parser.add_argument("--top_k", type=int, help="Top most frequent words to leave")
  parser.add_argument("path", help="Path to file")
  parser.add_argument("type", type=str, default='text', help="The text to use")
  parser.add_argument("vocab", help="Destination")
  args = parser.parse_args()

  if args.type == 'text':
    vocab = Vocabulary.build(args.path, args.top_k)
  elif args.type == 'snli':
    pass
    # SNLI has special format
  else:
    raise NotImplementedError(args.type)

  vocab.save(args.vocab)

if __name__ == "__main__":
    main()
