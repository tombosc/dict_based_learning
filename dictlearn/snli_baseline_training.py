"""
Training loop for baseline SNLI model

TODO: Dropout circuit
TODO: Fix embeddings (shouldnt be trained according to keras_snli repo, this will also speed up code)
"""

import sys

import numpy as np
from theano import tensor

sys.path.append("..")

from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import apply_dropout, apply_batch_normalization
from blocks.algorithms import Scale
from blocks.extensions import ProgressBar, Timestamp

import os
import time
import atexit
import signal
import pprint
import logging
import cPickle
import subprocess

import json

import numpy
import theano
from theano import tensor as T

from blocks.algorithms import (
    GradientDescent, Adam)
from blocks.graph import ComputationGraph, apply_batch_normalization, apply_dropout, get_batch_normalization_updates
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing, first
from blocks.extensions.saveload import Load, Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

from blocks.roles import WEIGHT
from blocks.filter import VariableFilter

from fuel.streams import ServerDataStream

from dictlearn.util import get_free_port, configure_logger
from dictlearn.extensions import DumpTensorflowSummaries, SimpleExtension
from dictlearn.data import SNLIData
from dictlearn.snli_baseline_model import SNLIBaseline
from dictlearn.retrieval import Retrieval, Dictionary

import pandas as pd
from collections import defaultdict

class LoggerPrinting(SimpleExtension):
    """
    Prints log messages to the screen.

    TODO(kudkudak): This could be also echieved by redirecting stdout to specially prepared logger,
    but it doesn't work nicely with progressbar, and progress bar is cool :(

    ( Progress bar sends new write to logger then so log file would have new line for each update
    of progress bar )
    """
    def __init__(self, logger_name, **kwargs):
        self._logger = logging.getLogger(logger_name)
        self._logger_name = logger_name
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(LoggerPrinting, self).__init__(**kwargs)

    def __getstate__(self):
        # Ensure we won't pickle the actual progress bar.
        # (It might contain unpicklable file handles)
        state = dict(self.__dict__)
        del state['_logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._logger = logging.getLogger(logger_name)

    def _print_attributes(self, attribute_tuples):
        for attr, value in sorted(attribute_tuples.items(), key=first):
            if not attr.startswith("_"):
                self._logger.info("\t", "{}:".format(attr), value)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        print_status = True

        self._logger.info()
        self._logger.info("".join(79 * "-"))
        if which_callback == "before_epoch" and log.status['epochs_done'] == 0:
            self._logger.info("BEFORE FIRST EPOCH")
        elif which_callback == "on_resumption":
            self._logger.info("TRAINING HAS BEEN RESUMED")
        elif which_callback == "after_training":
            self._logger.info("TRAINING HAS BEEN FINISHED:")
        elif which_callback == "after_epoch":
            self._logger.info("AFTER ANOTHER EPOCH")
        elif which_callback == "on_interrupt":
            self._logger.info("TRAINING HAS BEEN INTERRUPTED")
            print_status = False
        self._logger.info("".join(79 * "-"))
        if print_status:
            self._logger.info("Training status:")
            self._print_attributes(log.status)
            self._logger.info("Log records from the iteration {}:".format(
                log.status['iterations_done']))
            self._print_attributes(log.current_row)
        self._logger.info()


class DumpCSVSummaries(SimpleExtension):
    def __init__(self, save_path, mode="w", **kwargs):
        self._save_path = save_path
        self._mode = mode

        if self._mode == "w":
            # Clean up file
            with open(os.path.join(self._save_path, "logs.csv"), "w") as f:
                pass
            self._current_log = defaultdict(list)
        else:
            self._current_log = pd.read_csv(os.path.join(self._save_path, "logs.csv")).to_dict()

        super(DumpCSVSummaries, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        for key, value in self.main_loop.log.current_row.items():
            try:
                float_value = float(value)
                self._current_log[key].append(float_value)
            except:
                pass

        # Make sure all logs have same length (for csv serialization)
        max_len = max([len(v) for v in self._current_log.values()])
        for k in self._current_log:
            if len(self._current_log[k]) != max_len:
                self._current_log[k] += [self._current_log[k][-1] for _ in range(max_len - len(self._current_log[k]))]

        pd.DataFrame(self._current_log).to_csv(os.path.join(self._save_path, "logs.csv"))

def train_snli_model(config, save_path, params, fast_start, fuel_server):
    c = config
    new_training_job = False
    logger = configure_logger(name="snli_baseline_training", log_file=os.path.join(save_path, "log.txt"))
    if not os.path.exists(save_path):
        logger.info("Start a new job")
        new_training_job = True
        os.mkdir(save_path)
    else:
        logger.info("Continue an existing job")
    main_loop_path = os.path.join(save_path, 'main_loop.tar')
    stream_path = os.path.join(save_path, 'stream.pkl')

    # Save config to save_path
    json.dump(config, open(os.path.join(save_path, "config.json"), "w"))

    # Load data
    data = SNLIData(c['data_path'], c['layout'])

    # Dict
    if c['dict_path']:
        dict = Dictionary(c['dict_path'])
        retrieval = Retrieval(vocab=data.vocab, dictionary=dict, max_def_length=c['max_def_length'],
            exclude_top_k=c['exclude_top_k'])
    else:
        retrieval = None

    # Initialize
    baseline = SNLIBaseline(
        emb_dim=c['emb_dim'], vocab=data.vocab, encoder=c['encoder'], dropout=c['dropout'],
        num_input_words=c['num_input_words'],
        # Dict lookup kwargs (will get refactored)
        translate_dim=c['translate_dim'], retrieval=retrieval, compose_type=c['compose_type'],
        disregard_word_embeddings=c['disregard_word_embeddings']
        )
    baseline.initialize()
    embeddings = np.load(c['embedding_path'])
    baseline.set_embeddings(embeddings.astype(theano.config.floatX))

    # Compute cost
    s1, s2 = T.lmatrix('sentence1_ids'), T.lmatrix('sentence2_ids')
    s1_mask, s2_mask = T.fmatrix('sentence1_ids_mask'), T.fmatrix('sentence2_ids_mask')
    y = T.ivector('label')
    pred = baseline.apply(s1, s1_mask, s2, s2_mask)
    cost = CategoricalCrossEntropy().apply(y.flatten(), pred)

    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4)
                .get_epoch_iterator())
        s1.tag.test_value = test_value_data[0]
        s1_mask.tag.test_value = test_value_data[1]
        s2.tag.test_value = test_value_data[2]
        s2_mask.tag.test_value = test_value_data[3]
        y.tag.test_value = test_value_data[4]

    # Computation graph
    cg = ComputationGraph([cost])

    # Weight decay
    weights = VariableFilter(bricks=[dense for dense, bn in baseline._mlp], roles=[WEIGHT])(cg.variables)
    final_cost = cost + np.float32(c['l2']) * sum((w ** 2).sum() for w in weights)
    final_cost.name = 'final_cost'

    for name, param, var in baseline.get_cg_transforms():
        logger.info("Applying " + name + " to " + var.name)
        cg = apply_dropout(cg, [var], param)

    cg = apply_batch_normalization(cg)
    # Add updates for population parameters
    pop_updates = get_batch_normalization_updates(cg)
    extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
        for p, m in pop_updates]

    # Optimizer
    algorithm = GradientDescent(
        cost=final_cost,
        parameters=cg.parameters,
        step_rule=Adam(learning_rate=c['lr']))
    algorithm.add_updates(extra_updates)
    m = Model(final_cost)

    # Monitors
    error_rate = MisclassificationRate().apply(y.flatten(), pred)

    parameters = m.get_parameter_dict()  # Blocks version mismatch
    logger.info("Trainable parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                        for key in sorted(parameters.keys())],
                    width=120))
    logger.info("Parameter norms" + "\n" +
                pprint.pformat(
                    [(key, np.linalg.norm(parameters[key].get_value().reshape(-1,)).mean())
                        for key in sorted(parameters.keys())],
                    width=120))

    train_monitored_vars = [final_cost, cost, error_rate]
    monitored_vars = [final_cost, cost, error_rate]
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
        Timestamp(),
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
        DumpCSVSummaries(
            save_path,
            every_n_batches=c['mon_freq_train'],
            after_training=True),
        LoggerPrinting(logger_name=logger.name, every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches'])
    ]

    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'],
        seed=numpy.random.randint(0, 10000000))

    if fuel_server:
        with open(stream_path, 'w') as dst:
            cPickle.dump(training_stream, dst, 0)
        port = ServerDataStream.PORT = get_free_port()
        ret = subprocess.Popen([os.path.join(os.path.dirname(__file__), "../bin/start_fuel_server.py"),
            stream_path, str(port)])
        print("Using port " + str(port))
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))
        training_stream = ServerDataStream(
            sources=training_stream.sources,
            produces_examples=training_stream.produces_examples, port=port)

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()
