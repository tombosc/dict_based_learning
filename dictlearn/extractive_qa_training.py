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
from dictlearn.data import ExtractiveQAData
from dictlearn.extensions import DumpTensorflowSummaries
from dictlearn.extractive_qa_model import ExtractiveQAModel
from dictlearn.retrieval import Retrieval, Dictionary

logger = logging.getLogger()


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
    main_loop_path = os.path.join(save_path, 'main_loop.tar')

    c = config
    data = ExtractiveQAData(c['data_path'], c['layout'])

    qam = ExtractiveQAModel(c['dim'], c['emb_dim'], c['num_input_words'],
                            data.vocab,
                            weights_init=Uniform(width=0.1),
                            biases_init=Constant(0.))
    qam.initialize()
    if c['embedding_path']:
        qam.set_embeddings(numpy.load(c['embedding_path']))

    contexts = tensor.ltensor3('contexts')
    context_mask = tensor.matrix('contexts_mask')
    questions = tensor.ltensor3('questions')
    question_mask = tensor.matrix('questions_mask')
    answer_begins = tensor.lvector('answer_begins')
    answer_ends = tensor.lvector('answer_ends')
    input_vars = [contexts, context_mask,
                  questions, question_mask,
                  answer_begins, answer_ends]

    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4, max_length=5)
            .get_epoch_iterator(as_dict=True))
        for var in input_vars:
            var.tag.test_value = test_value_data[var.name]

    costs = qam.apply(*input_vars)
    cost = rename(costs.mean(), 'mean_cost')

    cg = Model(cost)
    if params:
        logger.debug("Load parameters from {}".format(params))
        with open(params) as src:
            cg.set_parameter_values(load_parameters(src))

    length = rename(contexts.shape[1], 'length')
    exact_match, = VariableFilter(name='exact_match')(cg)
    exact_match_ratio = rename(exact_match.mean(), 'exact_match_ratio')
    monitored_vars = [length, cost, exact_match_ratio]

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
        for name, param in parameters.items():
            num_elements = numpy.product(param.get_value().shape)
            norm = param.norm(2) / num_elements ** 0.5
            grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
            step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
            stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
            stats.name = name + '_stats'
            train_monitored_vars.append(stats)

    extensions = [
        Load(main_loop_path, load_iteration_state=True, load_log=True)
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
                every_n_batches=c['mon_freq_valid']),
        Checkpoint(main_loop_path,
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
