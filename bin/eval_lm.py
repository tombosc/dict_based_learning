#!/usr/bin/env python
from dictlearn.language_model_training import initialize_data_and_model
from dictlearn.main import main_evaluate
from blocks.serialization import load_parameters
from blocks.model import Model
from blocks.filter import VariableFilter
from collections import Counter
import json
import theano
import theano.tensor as T

def evaluate_lm(config, tar_path, part, num_examples, dest_path, **kwargs):
    print "kwargs:", kwargs
    # pass in kwargs an additional
    # we ignore part, num_examples
    c = config

    data, lm, _ = initialize_data_and_model(c)
    words = T.ltensor3('words')
    words_mask = T.matrix('words_mask')

    costs = lm.apply(words, words_mask)
    cg = Model(costs)

    with open(tar_path) as src:
        cg.set_parameter_values(load_parameters(src))


    perplexities = VariableFilter(name_regex='perplexity.*')(cg)
    proba_out, = VariableFilter(name='proba_out')(cg)
    unk_ratios = VariableFilter(name_regex='def_unk_ratio.*')(cg)
    num_definitions, = VariableFilter(name='num_definitions')(cg)
    name_to_aggregate = [p.name for p in perplexities + [unk_ratios]]

    compute_l = perplexities + unk_ratios + [proba_out, num_definitions]
    compute = dict({p.name: p for p in compute_l})

    print "to compute:", compute.keys()

    predict_f = theano.function([words, words_mask], compute)

    stream = data.get_stream('validation', batch_size=50, max_length=100)
    raw_data = [] # list of dicts containing the inputs and computed outputs
    for input_data in stream.get_epoch_iterator(as_dict=True):
        words = input_data['words']
        words_mask = input_data['words_mask']
        to_save = predict_f(words_mask, words)
        to_save.update(input_data)
        raw_data.append(to_save)

    # aggregate
    aggregated = {}
    sum_mask_track = Counter()
    for d in raw_data:
        coef = d['words_mask'].sum()
        for name in name_to_aggregate:
            aggregated[name] += d[name] * coef
            sum_mask_track[name] += coef
    
    #TODO: check that different batch_size yields same validation error than 
    # end of training validation error.
    # TODO: I think blocks aggreg is simply mean which should break 
    # when we use masks??? investigate

    parts = ['test', 'test_unseen']
    stream = data.get_stream('test_unseen', batch_size=50, max_length=100)
    input_data = next(stream.get_epoch_iterator(as_dict=True))
    to_save = predict_f(input_data['words'], input_data['words_mask'])
    to_save.update(input_data)

    all_data = {'validation': aggregated, 'test_unseen': to_save} 
    json.dumps(all_data, dest_path)


if __name__ == "__main__":
    main_evaluate(lm_config_registry, evaluate_lm)
