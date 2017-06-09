#!/usr/bin/env bash
# Before running this script you have to add symlinks 
# train.json and dev.json to the training and development sets respectively
set -x

[ -e train.json ] || exit 1
[ -e dev.json ] || exit 1
[ -e glove.840B.300d.txt ] || exit 1 

[ -z "$CLASSPATH" ] && export CLASSPATH="$HOME/Dist/stanford-corenlp-full-2016-10-31/*"
[ -z $DBL ] && DBL=$HOME/Dist/dict_based_learning

cat >vocab.txt << EOM
<bos> 0
<eos> 0
<bod> 0
<eod> 0
<unk> 0
EOM

cut -d' ' -f1 glove.840B.300d.txt | awk '{print $1, 0}' >>vocab.txt
$DBL/bin/pack_glove.py glove.840B.300d.txt glove_w_specials.npy

# for CoreNLP
$DBL/bin/pack_to_hdf5.py --type=squad train.json train.h5
$DBL/bin/pack_to_hdf5.py --type=squad --relaxed-span dev.json dev.h5
