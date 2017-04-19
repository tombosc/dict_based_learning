import os
import time
import socket
import atexit
import signal
import pprint
import logging
import cPickle
import subprocess

import numpy
import theano
from theano import tensor

from blocks.initialization import Uniform, Constant
from blocks.algorithms import (
    Adam, GradientDescent, Adam, StepClipping, CompositeRule)
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Load, Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.serialization import load_parameters

from fuel.streams import ServerDataStream

from dictlearn.util import rename, masked_root_mean_square, get_free_port
from dictlearn.theano_util import parameter_stats
from dictlearn.data import LanguageModellingData
from dictlearn.extensions import (
    DumpTensorflowSummaries, StartFuelServer)
from dictlearn.language_model import LanguageModel
from dictlearn.retrieval import Retrieval, Dictionary

logger = logging.getLogger()


def train_language_model(config, save_path, params, fast_start, fuel_server):
    new_training_job = False
    if not os.path.exists(save_path):
        logger.info("Start a new job")
        new_training_job = True
        os.mkdir(save_path)
    else:
        logger.info("Continue an existing job")
    main_loop_path = os.path.join(save_path, 'main_loop.tar')
    stream_path = os.path.join(save_path, 'stream.pkl')

    c = config
    data = LanguageModellingData(c['data_path'], c['layout'])
    retrieval = None
    if c['dict_path']:
        retrieval = Retrieval(data.vocab, Dictionary(c['dict_path']),
                              c['max_def_length'], c['exclude_top_k'])

    lm = LanguageModel(c['dim'], c['num_input_words'], c['num_output_words'],
                       data.vocab, retrieval,
                       c['standalone_def_rnn'],
                       c['disregard_word_embeddings'],
                       c['compose_type'],
                       weights_init=Uniform(width=0.1),
                       biases_init=Constant(0.))
    lm.initialize()

    words = tensor.ltensor3('words')
    words_mask = tensor.matrix('words_mask')
    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4, max_length=5)
            .get_epoch_iterator())
        words.tag.test_value = test_value_data[0]
        words_mask.tag.test_value = test_value_data[1]

    costs = lm.apply(words, words_mask)
    cost = rename(costs.mean(), 'mean_cost')

    cg = Model(cost)
    if params:
        logger.debug("Load parameters from {}".format(params))
        with open(params) as src:
            cg.set_parameter_values(load_parameters(src))


    length = rename(words.shape[1], 'length')
    last_correct, = VariableFilter(name='last_correct')(cg)
    last_correct_acc = rename(last_correct.mean(), 'last_correct_acc')
    perplexity, = VariableFilter(name='perplexity')(cg)
    monitored_vars = [length, cost, last_correct_acc, perplexity]
    if c['dict_path']:
        num_definitions, = VariableFilter(name='num_definitions')(cg)
        max_definition_length, = VariableFilter(name='max_definition_length')(cg)
        monitored_vars.extend([num_definitions, max_definition_length])

    parameters = cg.get_parameter_dict()
    logger.info("Trainable parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                     for key in sorted(parameters.keys())],
                    width=120))

    rules = []
    if c['grad_clip_threshold']:
        rules.append(StepClipping(c['grad_clip_threshold']))
    rules.append(Adam(learning_rate=c['learning_rate'],
                      beta1=c['momentum']))
    algorithm = GradientDescent(
        cost=cost,
        parameters=parameters.values(),
        step_rule=CompositeRule(rules))
    train_monitored_vars = list(monitored_vars)
    if c['grad_clip_threshold']:
        train_monitored_vars.append(algorithm.total_gradient_norm)

    if c['dict_path']:
        def_mean_rootmean2, = VariableFilter(name='def_mean_rootmean2')(cg)
        merged_input_rootmean2, = VariableFilter(name='merged_input_rootmean2')(cg)
        train_monitored_vars.extend([def_mean_rootmean2, merged_input_rootmean2])
    rnn_input_rootmean2, = VariableFilter(name='rnn_input_rootmean2')(cg)
    train_monitored_vars.append(rnn_input_rootmean2)

    if 0:
        input_gates, = VariableFilter(bricks=[lm._main_rnn], name='input_gates')(cg)
        forget_gates, = VariableFilter(bricks=[lm._main_rnn], name='forget_gates')(cg)
        output_gates, = VariableFilter(bricks=[lm._main_rnn], name='output_gates')(cg)
        train_monitored_vars.append(
            rename(masked_root_mean_square(input_gates, words_mask.T), 'input_gate_root_mean2'))
        train_monitored_vars.append(
            rename(masked_root_mean_square(forget_gates, words_mask.T), 'forget_gate_root_mean2'))
        train_monitored_vars.append(
            rename(masked_root_mean_square(output_gates, words_mask.T), 'output_gate_root_mean2'))
    if 1:
        main_rnn_states = VariableFilter(applications=[lm._main_rnn.apply], name='states')(cg)[-1]
        train_monitored_vars.append(
            rename(masked_root_mean_square(main_rnn_states, words_mask.T), 'main_rnn_states_root_mean2'))

    if c['monitor_parameters']:
        train_monitored_vars.extend(parameter_stats(parameters, algorithm))

    # We use a completely random seed on purpose. With Fuel server
    # it's currently not possible to restore the state of the training
    # stream. That's why it's probably better to just have it stateless.
    stream_seed = numpy.random.randint(0, 10000000) if fuel_server else None
    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'], max_length=c['max_length'],
        seed=stream_seed)
    original_training_stream = training_stream
    if fuel_server:
        # the port will be configured by the StartFuelServer extension
        training_stream = ServerDataStream(
            sources=training_stream.sources,
            produces_examples=training_stream.produces_examples)

    extensions = [
        Load(main_loop_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job),
        StartFuelServer(original_training_stream,
                        stream_path,
                        before_training=fuel_server),
        Timing(every_n_batches=c['mon_freq_train']),
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
        DataStreamMonitoring(
            monitored_vars,
            data.get_stream('valid', batch_size=c['batch_size_valid']),
            prefix="valid").set_conditions(
                before_first_epoch=not fast_start,
                every_n_batches=c['mon_freq_valid']),
        Checkpoint(main_loop_path,
                   save_separately=['iteration_state'],
                   before_training=not fast_start,
                   every_n_batches=c['save_freq_batches'],
                   after_training=not fast_start),
        DumpTensorflowSummaries(
            save_path,
            every_n_batches=c['mon_freq_train'],
            after_training=True),
        Printing(every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches'])
    ]

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()
