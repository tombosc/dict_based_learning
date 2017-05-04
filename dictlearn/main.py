#!/usr/bin/env python
import pprint
import argparse

import logging
logging.basicConfig(
    level='DEBUG',
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def add_config_arguments(config, parser):
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(
                "--" + key, dest=key, default=None, action="store_true",
                help="Enable a setting from the configuration")
            parser.add_argument(
                "--no_" + key, dest=key, default=None, action="store_false",
                help="Disable a setting from the configuration")
        else:
            parser.add_argument(
                "--" + key, type=type(value),
                help="A setting from the configuration")


def main(config_registry, training_func):
    parser = argparse.ArgumentParser("Learning with a dictionary")
    parser.add_argument("--fast-start", action="store_true",
                        help="Start faster by skipping a few things at the start")
    parser.add_argument("--fuel-server", action="store_true",
                        help="Use standalone Fuel dataset server")
    parser.add_argument("--params",
                        help="Load parameters from a main loop")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("save_path", help="The destination for saving")
    add_config_arguments(config_registry.get_root_config(), parser)

    args = parser.parse_args()

    # Modify the configuration with the command line arguments
    config = config_registry[args.config]
    for key in config:
        if key in args and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    pprint.pprint(config)

    # For now this script just runs the language model training.
    # More stuff to come.
    training_func(config, args.save_path,
                  args.params, args.fast_start, args.fuel_server)


def main_evaluate(config_registry, evaluate_func):
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument("--part", default='train', help="Part")
    parser.add_argument("--dest", help="Destination for outputs")
    parser.add_argument("--num-examples", type=int, help="Number of examples to read")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("tar_path", help="The tar file with parameters")
    add_config_arguments(config_registry.get_root_config(), parser)

    args = parser.parse_args()

    # Modify the configuration with the command line arguments
    config = config_registry[args.config]
    for key in config:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    pprint.pprint(config)

    # For now this script just runs the language model training.
    # More stuff to come.
    evaluate_func(config, args.tar_path, args.part, args.num_examples, args.dest)
