#!/usr/bin/env python

import argparse
import cPickle
import logging

from fuel.server import start_server

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser("Builds a dictionary")
    parser.add_argument("stream", help="The path to the pickled stream")
    parser.add_argument("port", type=int, help="The port to use")
    parser.add_argument("hwm", type=int, default=10, help="HWM")
    args = parser.parse_args()
    stream = cPickle.load(open(args.stream))
    start_server(stream, args.port, hwm=args.hwm)


if __name__ == "__main__":
    main()
