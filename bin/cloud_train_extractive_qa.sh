#!/usr/bin/env bash
set -x

export

export PYTHONUNBUFFERED=1
export THEANO_FLAGS="device=cuda,compiledir=/workspace/.theano"
export PYTHONPATH=/workspace/Theano:/workspace/blocks:/workspace/fuel:/workspace/dict_based_learning
export NLTK_DATA=/workspace/nltk_data
export PATH=$PATH:/workspace/dict_based_learning/bin

/workspace/dict_based_learning/bin/train_extractive_qa.py $@