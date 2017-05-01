#!/usr/bin/env bash
set -x

export PYTHONUNBUFFERED=1
export THEANO_FLAGS="device=gpu,compiledir=/workspace/.theano"
export PYTHONPATH=/workspace/blocks:/workspace/fuel:/workspace/dict_based_learning
export NLTK_DATA=/workspace/nltk_data

/workspace/dict_based_learning/bin/train_extractive_qa.py --data_path /data/cf9ffb48-61bd-40dc-a011-b2e7e5acfd72/squad/squad_from_scratch $@
