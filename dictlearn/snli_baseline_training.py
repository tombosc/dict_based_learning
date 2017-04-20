"""
Training loop for baseline SNLI model

TODO: Debug low acc
TODO: Unit test data preprocessing
TODO: Add logging to txt
TODO: Reload with fuel server?
TODO: Second round of debugging reloading
TODO: Add assert that embeddings are frozen

Diff:
BN during training using sample stat
"""

import sys

import numpy as np
from theano import tensor

sys.path.append("..")

from blocks.bricks.cost import MisclassificationRate
from blocks.filter import get_brick
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import apply_dropout, apply_batch_normalization
from blocks.algorithms import Scale, RMSProp
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

from fuel.streams import ServerDataStream, AbstractDataStream, zmq, recv_arrays
from subprocess import Popen, PIPE

from dictlearn.util import get_free_port, configure_logger, copy_streams_to_file
from dictlearn.extensions import DumpTensorflowSummaries, SimpleExtension, StartFuelServer
from dictlearn.data import SNLIData
from dictlearn.snli_baseline_model import SNLIBaseline
from dictlearn.retrieval import Retrieval, Dictionary

import pandas as pd
from collections import defaultdict


class ServerDataStream(AbstractDataStream):
    """A data stream that receives batches from a Fuel server.

    Parameters
    ----------
    sources : tuple of strings
        The names of the data sources returned by this data stream.
    produces_examples : bool
        Whether this data stream produces examples (as opposed to batches
        of examples).
    host : str, optional
        The host to connect to. Defaults to ``localhost``.
    port : int, optional
        The port to connect on. Defaults to 5557.
    hwm : int, optional
        The `ZeroMQ high-water mark (HWM)
        <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
        receiving socket. Increasing this increases the buffer, which can
        be useful if your data preprocessing times are very random.
        However, it will increase memory usage. There is no easy way to
        tell how many batches will actually be queued with a particular
        HWM. Defaults to 10. Be sure to set the corresponding HWM on the
        server's end as well.
    axis_labels : dict, optional
        Maps source names to tuples of strings describing axis semantics,
        one per axis. Defaults to `None`, i.e. no information is available.

    """

    def __init__(self, sources, produces_examples, host='localhost', port=5557,
            hwm=10, axis_labels=None):
        super(ServerDataStream, self).__init__(axis_labels=axis_labels)
        self.sources = sources
        self.produces_examples = produces_examples
        self.host = host
        self.port = port
        self.hwm = hwm
        self.connect()

    def connect(self):
        context = zmq.Context()
        self.socket = socket = context.socket(zmq.PULL)
        socket.set_hwm(self.hwm)
        socket.connect("tcp://{}:{}".format(self.host, self.port))
        self.connected = True

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.connected:
            self.connect()
        data = recv_arrays(self.socket)
        return tuple(data)

    def get_epoch_iterator(self, **kwargs):
        return super(ServerDataStream, self).get_epoch_iterator(**kwargs)

    def close(self):
        pass

    def next_epoch(self):
        pass

    def reset(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['connected'] = False
        if 'socket' in state:
            del state['socket']
        return state


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
    with open(os.path.join(save_path, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv))
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
        disregard_word_embeddings=c['disregard_word_embeddings'], multimod_drop=c['multimod_drop']
    )
    baseline.initialize()

    if c['embedding_path']:
        embeddings = np.load(c['embedding_path'])
        baseline.set_embeddings(embeddings.astype(theano.config.floatX))

    # Compute cost
    s1, s2 = T.lmatrix('sentence1_ids'), T.lmatrix('sentence2_ids')
    s1_words, s2_words = T.lmatrix('sentence1_ids'), T.lmatrix('sentence2_ids')
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

    # Monitors
    error_rate = MisclassificationRate().apply(y.flatten(), pred)

    # Computation graph
    cg = ComputationGraph([cost, error_rate])

    # Weight decay (TODO: Make it less bug prone)
    weights = VariableFilter(bricks=[dense for dense, relu, bn in baseline._mlp], roles=[WEIGHT])(cg.variables)
    final_cost = cost + np.float32(c['l2']) * sum((w ** 2).sum() for w in weights)
    final_cost.name = 'final_cost'

    cg = apply_batch_normalization(cg)
    # Add updates for population parameters
    pop_updates = get_batch_normalization_updates(cg)
    extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
        for p, m in pop_updates]

    # extra_updates = []
    # pop_updates = []

    test_cg = cg
    for name, param, var in baseline.get_cg_transforms():
        logger.info("Applying " + name + " to " + str(var))
        cg = apply_dropout(cg, [var], param)

    # Freeze embeddings
    if not c['train_emb']:
        frozen_params = [p for E in baseline.get_embeddings_lookups() for p in E.parameters]
    else:
        frozen_params = []
    train_params = [p for p in cg.parameters if p not in frozen_params]
    train_params_keys = [get_brick(p).get_hierarchical_name(p) for p in train_params]

    # Optimizer
    algorithm = GradientDescent(
        cost=final_cost,
        on_unused_sources='ignore',
        parameters=train_params,
        step_rule=Adam(learning_rate=c['lr']))
    algorithm.add_updates(extra_updates)
    m = Model(final_cost)

    parameters = m.get_parameter_dict()  # Blocks version mismatch
    logger.info("Trainable parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                        for key in sorted(train_params_keys)],
                    width=120))
    logger.info("# of parameters {}".format(
        sum([np.prod(parameters[key].get_value().shape) for key in sorted(train_params_keys)])))
    logger.info("Parameter norms" + "\n" +
                pprint.pformat(
                    [(key, np.linalg.norm(parameters[key].get_value().reshape(-1, )).mean())
                        for key in sorted(train_params_keys)],
                    width=120))

    train_monitored_vars = [final_cost] + cg.outputs
    monitored_vars = test_cg.outputs
    if c['monitor_parameters']:
        for name in train_params_keys:
            param = parameters[name]
            num_elements = numpy.product(param.get_value().shape)
            norm = param.norm(2) / num_elements
            grad_norm = algorithm.gradients[param].norm(2) / num_elements
            step_norm = algorithm.steps[param].norm(2) / num_elements
            stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
            stats.name = name + '_stats'
            train_monitored_vars.append(stats)

    regular_training_stream = data.get_stream(
        'train', batch_size=c['batch_size'],
        seed=numpy.random.randint(0, 10000000))

    if fuel_server:
        # the port will be configured by the StartFuelServer extension
        training_stream = ServerDataStream(
            sources=regular_training_stream.sources,
            produces_examples=regular_training_stream.produces_examples)
    else:
        training_stream = regular_training_stream

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
            data.get_stream('valid', batch_size=c['batch_size']),
            prefix="valid").set_conditions(
            before_training=not fast_start,
            every_n_batches=c['mon_freq_valid']),
        # DataStreamMonitoring(
        #     monitored_vars,
        #     data.get_stream('test', batch_size=c['batch_size']),
        #     after_training=True,
        #     prefix="test"),
        Checkpoint(main_loop_path,
            parameters=cg.parameters + [p for p, m in pop_updates],
            before_training=not fast_start,
            every_n_batches=c['save_freq_batches'],
            after_training=not fast_start),
        # DumpTensorflowSummaries(
        #     save_path,
        #     every_n_batches=c['mon_freq_train'],
        #     after_training=True),
        # AutomaticKerberosCall(
        #     every_n_batches=c['mon_freq_train']),
        StartFuelServer(regular_training_stream,
            stream_path,
            before_training=fuel_server),
        DumpCSVSummaries(
            save_path,
            every_n_batches=c['mon_freq_train'],
            after_training=True),
        Printing(every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches'])
    ]

    if "VISDOM_SERVER" in os.environ:
        print("Running visdom server")
        ret = subprocess.Popen([os.path.join(os.path.dirname(__file__), "../visdom_plotter.py"),
            "--visdom-server={}".format(os.environ['VISDOM_SERVER']), "--folder={}".format(save_path)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))

    model = Model(cost)
    for p, m in pop_updates:
        model._parameter_dict[get_brick(p).get_hierarchical_name(p)] = p

    if c['embedding_path']:
        assert np.all(baseline.get_embeddings_lookups()[0].parameters[0].get_value(0)[123] == embeddings[123])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=model,
        extensions=extensions)

    assert os.path.exists(save_path)
    main_loop.run()
