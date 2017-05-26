#!/usr/bin/env python
import os
from dictlearn.language_model_training import initialize_data_and_model
from dictlearn.obw_configs import lm_config_registry
from dictlearn.main import main_evaluate
from blocks.serialization import load_parameters
from blocks.model import Model
from blocks.filter import VariableFilter
from collections import Counter
import numpy as np
import json
import theano
import theano.tensor as T

def evaluate_lm(config, tar_path, part, num_examples, dest_path, **kwargs):
    c = config

    if part not in ['test_unseen', 'test']:
        raise ValueError()

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
    #num_definitions, = VariableFilter(name='num_definitions')(cg)
    print perplexities
    name_to_aggregate = [p.name for p in perplexities + unk_ratios]

    compute_l = perplexities + unk_ratios
    if part == 'test_unseen':
        compute_l.append(proba_out)

    compute = dict({p.name: p for p in compute_l})
    print "to compute:", compute.keys()
    predict_f = theano.function([words, words_mask], compute)

    batch_size = 50 # size of test_unseen
    stream = data.get_stream(part, batch_size=batch_size, max_length=100)
    raw_data = [] # list of dicts containing the inputs and computed outputs
    i=0
    print "start computing"
    for input_data in stream.get_epoch_iterator(as_dict=True):
        if i and i%10==0:
            print "iteration:", i
        words = input_data['words']
        words_mask = input_data['words_mask']
        to_save = predict_f(words, words_mask)
        if part == 'test_unseen':
            to_save.update(input_data)
        raw_data.append(to_save)
        i+=1

    # aggregate
    aggregated = Counter()
    sum_mask_track = Counter()
    for d in raw_data:
        coef = d['words_mask'].sum()
        for name in name_to_aggregate:
            aggregated[name] += d[name] * coef
            sum_mask_track[name] += coef

    for k,v in aggregated.iteritems():
        aggregated[k] = v/sum_mask_track[k]

    n_params = sum([np.prod(p.shape.eval()) for p in cg.parameters])
    aggregated['n_params'] = n_params
    print "# of parameters {}".format(n_params)


    #TODO: check that different batch_size yields same validation error than 
    # end of training validation error.
    # TODO: I think blocks aggreg is simply mean which should break 
    # when we use masks??? investigate

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if part == 'test_unseen':
        np.savez(os.path.join(dest_path, "predictions"),
             words = input_data['words'], 
             words_mask = input_data['words_mask'],
             proba_out = to_save['languagemodel_apply_proba_out'])

    json.dump(aggregated, 
            open(os.path.join(dest_path, "aggregates.json"), "w"),
            sort_keys=True, indent=2)

if __name__ == "__main__":
    main_evaluate(lm_config_registry, evaluate_lm)
