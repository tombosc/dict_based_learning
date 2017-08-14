set -x

export NLTK_DATA=/workspace/nltk_data 
export PYTHONIOENCODING=utf-8 
export PYTHONPATH=/workspace/dict_based_learning 
export FUEL_DATA_PATH=data 
export FUEL_FLOATX=float32 
export THEANO_FLAGS=optimizer=fast_run,floatX=float32
export CLASSPATH=/workspace/stanford-corenlp-full-2016-10-31/*

mkdir -p data/squad
ln -s $HOME/squad_data/squad_from_scratch data/squad/
ln -s $HOME/squad_data/squad_glove data/squad

JSON=$1
shift

/workspace/dict_based_learning/bin/pack_to_hdf5.py --type=squad --relaxed-span $JSON test.h5
/workspace/dict_based_learning/bin/eval_extractive_qa.py --dataset test.h5 $@ model/training_state_best.tar
