#!/bin/bash -x
# Preprocessing of SNLI dataset to h5 files, vocab.txt and vocab_all.txt

set -x

if [ -z "$DATA_DIR" ]; then
    echo "Need to set $DATA_DIR"
    exit 1
fi

# Assumes SNLI is download to $DATA_DIR/raw/snli_1.0

# Convert to h5 files
python bin/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_train.txt $DATA_DIR/snli/train.h5 --type=snli
python bin/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_dev.txt $DATA_DIR/snli/valid.h5 --type=snli
python bin/pack_to_hdf5.py $DATA_DIR/raw/snli_1.0/snli_1.0_test.txt $DATA_DIR/snli/test.h5 --type=snli

# Build vocab for both train and all data
python bin/build_vocab.py $DATA_DIR/snli/train.h5 $DATA_DIR/snli/vocab.txt
python bin/build_vocab.py $DATA_DIR/snli/train.h5,$DATA_DIR/snli/valid.h5,$DATA_DIR/snli/test.h5 $DATA_DIR/snli/vocab_all.txt