#!/usr/bin/env bash
set -x

export PYTHONUNBUFFERED=1
export THEANO_FLAGS="device=cuda,compiledir=/workspace/.theano"
export PYTHONPATH=/workspace/Theano:/workspace/blocks:/workspace/fuel:/workspace/dict_based_learning
export FUEL_DATA_PATH=/data/cf9ffb48-61bd-40dc-a011-b2e7e5acfd72
export NLTK_DATA=/workspace/nltk_data
export PATH=$PATH:/workspace/dict_based_learning/bin

JOBID=${MARATHON_APP_ID:1}

/workspace/dict_based_learning/bin/train_extractive_qa.py $@
