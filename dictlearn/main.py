
#!/usr/bin/env python
import pprint
import argparse
import logging


def main(config_registry, training_func):
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Learning with a dictionary")
    parser.add_argument("--fast-start", action="store_true",
                        help="Start faster by skipping a few things at the start")
    parser.add_argument("--fuel-server", action="store_true",
                        help="Use standalone Fuel dataset server")
    parser.add_argument("--params",
                        help="Load parameters from a main loop")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("save_path", help="The destination for saving")

    # Add all configuration options to the command line parser
    root_config = config_registry.get_root_config()
    for key, value in root_config.items():
        if isinstance(value, bool):
            parser.add_argument(
                "--" + key, dest=key, default=value, action="store_true",
                help="Enable a setting from the configuration")
            parser.add_argument(
                "--no_" + key, dest=key, default=value, action="store_false",
                help="Disable a setting from the configuration")
        else:
            parser.add_argument(
                "--" + key, type=type(value),
                help="A setting from the configuration")

    args = parser.parse_args()

    # Modify the configuration with the command line arguments
    config = config_registry[args.config]
    for key in config:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    pprint.pprint(config)

    # For now this script just runs the language model training.
    # More stuff to come.
    training_func(config, args.save_path,
                  args.params, args.fast_start, args.fuel_server)
