#!/usr/bin/env bash

set -x

export PYTHONIOENCODING=utf-8 
export PYTHONPATH=$HOME/dist/dict_based_learning:$HOME/dist/blocks:$HOME/dist/fuel:$HOME/dist/Theano
export FUEL_DATA_PATH=$HOME/data 
export FUEL_FLOATX=float32 
export THEANO_FLAGS=device=cuda,optimizer=fast_run,floatX=float32

$@
