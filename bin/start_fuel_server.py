#!/usr/bin/env python


import argparse
import cPickle
import logging

from fuel.server import start_server

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser("Starts fuel server")
    parser.add_argument("stream", help="The path to the pickled stream")
    parser.add_argument("port", type=int, help="The port to use")
    parser.add_argument("hwm", type=int, default=10, help="HWM")
    args = parser.parse_args()
    stream = cPickle.load(open(args.stream))
    try:
        start_server(stream, args.port, hwm=args.hwm)
    except KeyboardInterrupt:
        logger.info("Thank you for using Fuel server, bye-bye!")



if __name__ == "__main__":
    main()
