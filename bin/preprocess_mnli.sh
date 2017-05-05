#!/bin/bash -x
# Preprocessing of MSNLI dataset to h5 files, vocab.txt and vocab_all.txt

set -x

if [ -z "$DATA_DIR" ]; then
    echo "Need to set $DATA_DIR"
    exit 1
fi

# Assumes SNLI is download to $DATA_DIR/raw/multinli_0.9

# Convert to h5 files
python bin/pack_to_hdf5.py $DATA_DIR/raw/multinli_0.9/multinli_0.9_train.txt $DATA_DIR/mnli/train.h5 --type=snli
python bin/pack_to_hdf5.py $DATA_DIR/raw/multinli_0.9/multinli_0.9_dev_matched.txt $DATA_DIR/mnli/valid.h5 --type=snli
python bin/pack_to_hdf5.py $DATA_DIR/raw/multinli_0.9/multinli_0.9_dev_mismatched.txt $DATA_DIR/mnli/valid_mismatched.h5 --type=snli


# Build vocab for both train and all data
python bin/build_vocab.py $DATA_DIR/mnli/train.h5 $DATA_DIR/mnli/vocab.txt
python bin/build_vocab.py $DATA_DIR/mnli/train.h5,$DATA_DIR/mnli/valid_matched.h5,$DATA_DIR/mnli/valid_mismatched.h5 $DATA_DIR/mnli/vocab_all.txt