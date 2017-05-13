#!/usr/bin/env bash
set -x

source /workspace/dict_based_learning/bin/cloud_env.sh
python /workspace/dict_based_learning/bin/train_nli_esim.py $@ 2>$JOBID.txt 1>&2
