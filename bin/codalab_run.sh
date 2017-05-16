set -x

export NLTK_DATA=/workspace/nltk_data 
export PYTHONIOENCODING=utf-8 
export PYTHONPATH=/workspace/dict_based_learning 
export FUEL_DATA_PATH=data 
export FUEL_FLOATX=float32 
export THEANO_FLAGS=floatX=float32
export CLASSPATH=/workspace/stanford-corenlp-full-2016-10-31/*

mkdir -p data/squad
ln -s squad_from_scratch data/squad/
ln -s squad_glove data/squad

/workspace/dict_based_learning/bin/eval_extractive_qa.py --part dev $@ model/training_state_best.tar
