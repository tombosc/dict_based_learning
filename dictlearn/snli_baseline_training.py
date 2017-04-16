"""
Training loop for baseline SNLI model
"""

import sys

import numpy as np
from theano import tensor

sys.path.append("..")

from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import apply_dropout, apply_batch_normalization
from blocks.algorithms import Scale
from blocks.extensions import ProgressBar

import os
import time
import atexit
import signal
import pprint
import logging
import cPickle
import subprocess

import numpy
import theano
from theano import tensor as T

from blocks.algorithms import (
    GradientDescent)
from blocks.graph import ComputationGraph, apply_batch_normalization, apply_dropout, get_batch_normalization_updates
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Load, Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

from fuel.streams import ServerDataStream

from dictlearn.util import get_free_port
from dictlearn.extensions import DumpTensorflowSummaries
from dictlearn.data import SNLIData
from dictlearn.snli_baseline_model import SNLIBaseline

logger = logging.getLogger()


def train_snli_model(config, save_path, params, fast_start, fuel_server):
    c = config

    new_training_job = False
    if not os.path.exists(save_path):
        logger.info("Start a new job")
        new_training_job = True
        os.mkdir(save_path)
    else:
        logger.info("Continue an existing job")
    main_loop_path = os.path.join(save_path, 'main_loop.tar')
    stream_path = os.path.join(save_path, 'stream.pkl')

    # Load data
    data = SNLIData(c['data_path'], c['layout'])

    # Initialize
    baseline = SNLIBaseline(translate_dim=c['translate_dim'],
        emb_dim=c['emb_dim'], vocab=data.vocab, encoder=c['encoder'], dropout=c['dropout'])
    baseline.initialize()
    embeddings = np.load(c['embedding_path'])
    baseline.embeddings_var().set_value(embeddings.astype(theano.config.floatX))

    # Compute cost
    s1, s2 = T.lmatrix('sentence1_ids'), T.lmatrix('sentence2_ids')
    s1_mask, s2_mask = T.matrix('sentence1_ids_mask'), T.matrix('sentence2_ids_mask')
    y = T.ivector('label')
    pred = baseline.apply(s1, s1_mask, s2, s2_mask)
    cost = CategoricalCrossEntropy().apply(y.flatten(), pred)

    # Computation graph
    cg = ComputationGraph([cost])

    for name, param, var in baseline.get_cg_transforms():
        print("Applying " + name + " to " + var.name)
        cg = apply_dropout(cg, [var], param)

    cg = apply_batch_normalization(cg)
    # Add updates for population parameters
    pop_updates = get_batch_normalization_updates(cg)
    extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
        for p, m in pop_updates]

    # Optimizer
    algorithm = GradientDescent(
        cost=cost,
        parameters=cg.parameters,
        step_rule=Scale(learning_rate=c['lr']))
    algorithm.add_updates(extra_updates)
    m = Model(cost)

    # Monitors
    error_rate = MisclassificationRate().apply(y.flatten(), pred)

    parameters = m.get_parameter_dict()  # Blocks version mismatch
    logger.info("Trainable parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                        for key in sorted(parameters.keys())],
                    width=120))

    train_monitored_vars = [error_rate]
    monitored_vars = [error_rate]
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
        ProgressBar(),
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
        DataStreamMonitoring(
            monitored_vars,
            data.get_stream('dev', batch_size=c['batch_size']),
            prefix="dev").set_conditions(
            before_training=not fast_start,
            every_n_batches=c['mon_freq_dev']),
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

    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'],
        seed=numpy.random.randint(0, 10000000))

    if fuel_server:
        with open(stream_path, 'w') as dst:
            cPickle.dump(training_stream, dst, 0)
        port = ServerDataStream.PORT = get_free_port()
        ret = subprocess.Popen(["start_fuel_server.py", stream_path, str(port)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))
        training_stream = ServerDataStream(
            sources=training_stream.sources,
            produces_examples=training_stream.produces_examples)

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()
