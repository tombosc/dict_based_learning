#!/usr/bin/env python

import argparse
import cPickle

from fuel.server import start_server

def main():
  parser = argparse.ArgumentParser("Builds a dictionary")
  parser.add_argument("stream", help="The path to the pickled stream")
  parser.add_argument("port", type=int, help="The port to use")
  args = parser.parse_args()

  stream = cPickle.load(open(args.stream))
  start_server(stream, args.port)


if __name__ == "__main__":
    main()
