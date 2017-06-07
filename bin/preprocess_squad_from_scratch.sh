#!/usr/bin/env bash
# Before running this script you have to add symlinks 
# train.json and dev.json to the training and development sets respectively
set -x

[ -e train.json ] || exit 1
[ -e dev.json ] || exit 1

# for CoreNLP
export CLASSPATH=$HOME/Dist/stanford-corenlp-full-2016-10-31/* 

DBL=$HOME/Dist/dict_based_learning

$DBL/bin/pack_to_hdf5.py --type=squad train.json train.h5
$DBL/bin/pack_to_hdf5.py --type=squad --relaxed-span dev.json dev.h5

$DBL/bin/build_vocab.py train.h5 vocab.txt

