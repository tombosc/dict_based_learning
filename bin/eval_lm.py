#!/usr/bin/env python
from dictlearn.language_model_training import initialize_data_and_model
from dictlearn.obw_configs import lm_config_registry
from dictlearn.main import main_evaluate
from blocks.serialization import load_parameters
from blocks.model import Model
from blocks.filter import VariableFilter
import theano
import theano.tensor as T

def evaluate_lm(config, tar_path, part, num_examples, dest_path):
    c = config

    data, lm = initialize_data_and_model(c)
    words = T.ltensor3('words')
    words_mask = T.matrix('words_mask')

    costs = lm.apply(words, words_mask)
    cg = Model(costs)

    with open(tar_path) as src:
        cg.set_parameter_values(load_parameters(src))

    perplexities = VariableFilter(name_regex='perplexity.*')(cg)
    compute = {p.name: p for p in perplexities}
    #num_definitions, = VariableFilter(name='num_definitions')(cg)
    #compute['num_definitions'] = num_definitions
    print "to compute:", compute
    predict_f = theano.function([words, words_mask], compute)

    stream = data.get_stream(part, batch_size=1, seed=0, max_length=100)
    i = 0
    for example in stream.get_epoch_iterator(as_dict=True):
        print example['words_mask'], example['words']
        print predict_f(example['words'], example['words_mask'])
        i+=1
        if i > 100:
            break


if __name__ == "__main__":
    main_evaluate(lm_config_registry, evaluate_lm)
