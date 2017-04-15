from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from nltk.tokenize.moses import MosesDetokenizer

from blocks.initialization import Uniform, Constant
from blocks.algorithms import (
    Adam, GradientDescent, Adam, StepClipping, CompositeRule)
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.serialization import load_parameters

from dictlearn.util import rename, masked_root_mean_square, get_free_port
from dictlearn.theano_util import parameter_stats
from dictlearn.data import ExtractiveQAData
from dictlearn.extensions import DumpTensorflowSummaries, LoadNoUnpickling
from dictlearn.extractive_qa_model import ExtractiveQAModel
from dictlearn.retrieval import Retrieval, Dictionary

logger = logging.getLogger()


def _initialize_data_and_model(config):
    c = config
    data = ExtractiveQAData(c['data_path'], c['layout'])
    qam = ExtractiveQAModel(c['dim'], c['emb_dim'], c['num_input_words'],
                            data.vocab,
                            weights_init=Uniform(width=0.1),
                            biases_init=Constant(0.))
    qam.allocate()
    if c['embedding_path']:
        qam.set_embeddings(numpy.load(c['embedding_path']))
    logger.debug("Model created")
    return data, qam


def train_extractive_qa(config, save_path, params, fast_start, fuel_server):
    if fuel_server:
        raise NotImplementedError()

    new_training_job = False
    if not os.path.exists(save_path):
        logger.info("Start a new job")
        new_training_job = True
        os.mkdir(save_path)
    else:
        logger.info("Continue an existing job")
    tar_path = os.path.join(save_path, 'training_state.tar')

    c = config
    data, qam = _initialize_data_and_model(c)
    qam.initialize()

    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4, max_length=5)
            .get_epoch_iterator(as_dict=True))
        for var in qam.input_vars:
            var.tag.test_value = test_value_data[var.name]

    costs = qam.apply_with_default_vars()
    cost = rename(costs.mean(), 'mean_cost')

    cg = Model(cost)
    if params:
        logger.debug("Load parameters from {}".format(params))
        with open(params) as src:
            cg.set_parameter_values(load_parameters(src))

    length = rename(qam.contexts.shape[1], 'length')
    batch_size = rename(qam.contexts.shape[0], 'batch_size')
    exact_match, = VariableFilter(name='exact_match')(cg)
    exact_match_ratio = rename(exact_match.mean(), 'exact_match_ratio')
    context_word_ids, = VariableFilter(name='context_word_ids')(cg)
    num_unk = (tensor.eq(context_word_ids, data.vocab.unk) * qam.context_mask).sum()
    context_unk_ratio = rename(num_unk / qam.context_mask.sum(), 'context_unk_ratio')
    monitored_vars = [length, batch_size, cost, exact_match_ratio, context_unk_ratio]

    parameters = cg.get_parameter_dict()
    logger.info("Cost parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                     for key in sorted(parameters.keys())],
                    width=120))
    trained_parameters = parameters.values()
    if c['embedding_path']:
        logger.debug("Exclude  word embeddings from the trained parameters")
        trained_parameters = [p for p in trained_parameters
                              if not p == qam.embeddings_var()]

    rules = []
    if c['grad_clip_threshold']:
        rules.append(StepClipping(c['grad_clip_threshold']))
    rules.append(Adam(learning_rate=c['learning_rate'],
                      beta1=c['momentum']))
    algorithm = GradientDescent(
        cost=cost,
        parameters=trained_parameters,
        step_rule=CompositeRule(rules))
    train_monitored_vars = list(monitored_vars)
    if c['grad_clip_threshold']:
        train_monitored_vars.append(algorithm.total_gradient_norm)

    if c['monitor_parameters']:
        train_monitored_vars.extend(parameter_stats(parameters, algorithm))

    extensions = [
        LoadNoUnpickling(tar_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job),
        Timing(every_n_batches=c['mon_freq_train']),
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
        DataStreamMonitoring(
            monitored_vars,
            data.get_stream('dev', batch_size=c['batch_size_valid']),
            prefix="dev").set_conditions(
                before_training=not fast_start,
                after_epoch=True,
                every_n_batches=c['mon_freq_valid']),
        # We often use pretrained word embeddings and we don't want
        # to load and save them every time. To avoid that, we use
        # save_main_loop=False, we only save the trained parameters,
        # and we save the log and the iterations state separately
        # in the tar file.
        Checkpoint(tar_path,
                   parameters=trained_parameters,
                   save_main_loop=False,
                   save_separately=['log', 'iteration_state'],
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
    # We use a completely random seed on purpose. With Fuel server
    # it's currently not possible to restore the state of the training
    # stream. That's why it's probably better to just have it stateless.
    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'],
        shuffle=True, max_length=c['max_length'])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()


def evaluate_extractive_qa(config, save_path, part, dest):
    c = config
    data, qam = _initialize_data_and_model(c)
    costs = qam.apply_with_default_vars()
    cg = Model(costs)

    tar_path = os.path.join(save_path, 'training_state.tar')
    with open(tar_path) as src:
        cg.set_parameter_values(load_parameters(src))

    detok = MosesDetokenizer()
    def detokenize(str_):
        return " ".join(detok.detokenize(str_))

    predicted_begins, = VariableFilter(name='predicted_begins')(cg)
    predicted_ends, = VariableFilter(name='predicted_ends')(cg)
    predict_func = theano.function(qam.input_vars, [predicted_begins[0], predicted_ends[0]])

    num_examples = 0
    num_correct = 0
    def print_stats():
        print('EXACT MATCH RATIO: {}'.format(num_correct / float(num_examples)))

    stream = data.get_stream(part, batch_size=1, shuffle=part == 'train', raw_text=True)
    for example in stream.get_epoch_iterator(as_dict=True):
        feed = dict(example)
        feed['contexts'] = numpy.array(data.vocab.encode(example['contexts'][0]))[None, :]
        feed['questions'] = numpy.array(data.vocab.encode(example['questions'][0]))[None, :]
        correct_answer_span = slice(example['answer_begins'], example['answer_ends'])
        predicted_answer_span = slice(*predict_func(**feed))
        is_correct = correct_answer_span == predicted_answer_span

        num_examples += 1
        num_correct += is_correct

        result = 'correct' if is_correct else 'wrong'
        print('#{}'.format(num_examples))
        print("CONTEXT:", detokenize(example['contexts'][0]))
        print("QUESTION:", detokenize(example['questions'][0]))
        print("ANSWER (span=[{}, {}], {}):".format(predicted_answer_span.start,
                                                   predicted_answer_span.stop,
                                                   result),
              detokenize(example['contexts'][0, predicted_answer_span]))
        print()

        if num_examples % 100:
            print_stats()
    print_stats()
