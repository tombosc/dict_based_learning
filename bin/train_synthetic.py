#!/usr/bin/python

from dictlearn.language_model_training import train_language_model
import dictlearn.lm_configs_synthetic
from dictlearn.language_model_configs import lm_config_registry
import argparse

parser = argparse.ArgumentParser("Generate synthetic data and outputs in files")
parser.add_argument("results_path", type=str)
parser.add_argument("config_name", type=str)

args = parser.parse_args()

c = lm_config_registry[args.config_name]
train_language_model(c, args.results_path, None, True, False)
