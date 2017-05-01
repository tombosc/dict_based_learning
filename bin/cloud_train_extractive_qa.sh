#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export THEANO_FLAGS="device=gpu,compiledir=/workspace/.theano"
export PYTHONPATH=/workspace/blocks:/workspace/fuel:workspace/dict_based_learning
export NLTK_DATA=/workspace/nltk_data

/workspace/dict_based_learning/bin/train_extractive_qa.py $@
