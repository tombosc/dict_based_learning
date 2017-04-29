"""
Training loop for simple SNLI model that can use dict enchanced embeddings

TODO: What's best way to refactor it. I think training loop taking model + cost would be good enough for us
"""

import sys

import numpy as np
from theano import tensor

sys.path.append("..")

from blocks.bricks.cost import MisclassificationRate
from blocks.filter import get_brick
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.extensions import ProgressBar, Timestamp
from blocks.serialization import load_parameters
from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Uniform

from dictlearn.inits import GlorotUniform

import os
import time
import atexit
import signal
import pprint
import subprocess

import json

import numpy
import theano
from theano import tensor as T

from blocks.algorithms import (
    GradientDescent, Adam)
from blocks.graph import ComputationGraph, apply_batch_normalization, get_batch_normalization_updates
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Load, Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

from blocks.roles import WEIGHT
from blocks.filter import VariableFilter

from fuel.streams import ServerDataStream

from dictlearn.util import configure_logger
from dictlearn.extensions import StartFuelServer, DumpCSVSummaries, SimilarityWordEmbeddingEval, construct_embedder, \
    construct_dict_embedder
from dictlearn.data import SNLIData
from dictlearn.nli_simple_model import NLISimple
from dictlearn.retrieval import Retrieval, Dictionary


# class WeightedGradientDescent(GradientDescent):
#     """
#     Wraps GradientDescent adding shareable weights for updates
#     """
#     def __init__(self, parameters, **kwargs):
#
#         self._weight_dict = {}
#         for p in parameters:
#             self._weight_dict[p.name] = T.sharedvar()
#
#         super(WeightedGradientDescent, self).__init__(parameters=parameters, **kwargs)
#
#     def get_weight_dict(self):
#         pass
#
#     def _compute_gradients(self, known_grads, consider_constant):
#         gradients = super(WeightedGradientDescent, self).\
#             _compute_gradients(known_grads=known_grads, consider_constant=consider_constant)
#
#         return gradients



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
        dict = Dictionary(c['dict_path'], try_lowercase=c['try_lowercase'])
        retrieval = Retrieval(vocab=data.vocab, dictionary=dict, max_def_length=c['max_def_length'],
            exclude_top_k=c['exclude_top_k'], max_def_per_word=c['max_def_per_word'])
        retrieval_all = Retrieval(vocab=data.vocab, dictionary=dict, max_def_length=c['max_def_length'])
        data.set_retrieval(retrieval)
    else:
        retrieval = None

    # Initialize
    simple = NLISimple(
        # Common arguments
        emb_dim=c['emb_dim'], vocab=data.vocab, encoder=c['encoder'], dropout=c['dropout'],
        num_input_words=c['num_input_words'], mlp_dim=c['mlp_dim'],

        # Dict lookup kwargs (will get refactored)
        translate_dim=c['translate_dim'], retrieval=retrieval, compose_type=c['compose_type'],
        reader_type=c['reader_type'], disregard_word_embeddings=c['disregard_word_embeddings'],

        combiner_dropout=c['combiner_dropout'], share_def_lookup=c['share_def_lookup'],
        combiner_dropout_type=c['combiner_dropout_type'], combiner_bn=c['combiner_bn'],
        combiner_gating=c['combiner_gating'], combiner_shortcut=c['combiner_shortcut'],

        weights_init=GlorotUniform(), biases_init=Constant(0.0)
    )
    simple.push_initialization_config()
    if c['encoder'] == 'rnn':
        simple._rnn_encoder.weights_init = Uniform(std=0.1)
        # simple._rnn_fork.weights_init = Uniform(std=0.1)
    simple.initialize()

    if c['embedding_path']:
        embeddings = np.load(c['embedding_path'])
        simple.set_embeddings(embeddings.astype(theano.config.floatX))

    # Compute cost
    s1, s2 = T.lmatrix('sentence1'), T.lmatrix('sentence2')

    if c['dict_path']:
        s1_def_map, s2_def_map = T.lmatrix('sentence1_def_map'), T.lmatrix('sentence2_def_map')
        def_mask = T.fmatrix("def_mask")
        defs = T.lmatrix("defs")
    else:
        s1_def_map, s2_def_map = None, None
        def_mask = None
        defs = None

    s1_mask, s2_mask = T.fmatrix('sentence1_mask'), T.fmatrix('sentence2_mask')
    y = T.ivector('label')

    cg = {}
    for train_phase in [True, False]:
        pred = simple.apply(s1, s1_mask, s2, s2_mask, def_mask=def_mask, defs=defs, s1_def_map=s1_def_map,
            s2_def_map=s2_def_map, train_phase=train_phase)
        cost = CategoricalCrossEntropy().apply(y.flatten(), pred)
        error_rate = MisclassificationRate().apply(y.flatten(), pred)
        cg[train_phase] = ComputationGraph([cost, error_rate])
        cg[train_phase] = apply_batch_normalization(cg[train_phase])

    if params:
        logger.debug("Load parameters from {}".format(params))
        with open(params) as src:
            cg[True].set_parameter_values(load_parameters(src))

    # Weight decay (TODO: Make it less bug prone)
    weights_to_decay = VariableFilter(bricks=[dense for dense, relu, bn in simple._mlp], roles=[WEIGHT])(cg[True].variables)
    weight_decay = sum((w ** 2).sum() for w in weights_to_decay)
    final_cost = cg[True].outputs[0] + np.float32(c['l2']) * weight_decay
    final_cost.name = 'final_cost'

    # Add updates for population parameters
    pop_updates = get_batch_normalization_updates(cg[True])
    extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
        for p, m in pop_updates]

    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4)
                .get_epoch_iterator())
        s1.tag.test_value = test_value_data[0]
        s1_mask.tag.test_value = test_value_data[1]
        s2.tag.test_value = test_value_data[2]
        s2_mask.tag.test_value = test_value_data[3]
        y.tag.test_value = test_value_data[4]

    # TODO: Support freezing all but dict
    # Freeze embeddings
    if not c['train_emb']:
        frozen_params = [p for E in simple.get_embeddings_lookups() for p in E.parameters]
    else:
        frozen_params = []
    train_params = [p for p in cg[True].parameters if p not in frozen_params]
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

    train_monitored_vars = [final_cost] + cg[True].outputs
    monitored_vars = cg[False].outputs

    try:
        logger.info("Adding dict lookup norm tracking")
        train_monitored_vars.append(VariableFilter(name="s1_merged_input_rootmean2")(cg[True])[0])
        train_monitored_vars.append(VariableFilter(name="s1_def_mean_rootmean2")(cg[True])[0])
        monitored_vars.append(VariableFilter(name="s1_merged_input_rootmean2")(cg[False])[0])
        monitored_vars.append(VariableFilter(name="s1_def_mean_rootmean2")(cg[False])[0])
    except:
        pass

    try:
        logger.info("Adding gating tracking")
        train_monitored_vars.append(VariableFilter(name="s1_gate_rootmean2")(cg[True])[0])
        monitored_vars.append(VariableFilter(name="s1_gate_rootmean2")(cg[False])[0])
    except:
        pass

    try:
        logger.info("Adding gating tracking")
        train_monitored_vars.append(VariableFilter(name="s1_compose_gate_rootmean2")(cg[True])[0])
        monitored_vars.append(VariableFilter(name="s1_compose_gate_rootmean2")(cg[False])[0])
    except:
        pass

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
            sources=regular_training_stream.sources, hwm=100,
            produces_examples=regular_training_stream.produces_examples)
    else:
        training_stream = regular_training_stream

    extensions = [
        Load(main_loop_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job),
        StartFuelServer(regular_training_stream,
            stream_path,
            hwm=100,
            script_path=os.path.join(os.path.dirname(__file__), "../bin/start_fuel_server.py"),
            before_training=fuel_server),
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
        Checkpoint(main_loop_path,
            parameters=cg[True].parameters + [p for p, m in pop_updates],
            before_training=not fast_start,
            every_n_batches=c['save_freq_batches'],
            after_training=not fast_start)
    ]

    # Similarity trackers for embeddings
    for name in ['s1_word_embeddings', 's1_dict_word_embeddings', 's1_translated_word_embeddings']:
        variables = VariableFilter(name=name)(cg[False])
        if len(variables):
            print(variables)
            # TODO: Why is it 2?
            # assert len(variables) == 1, "Shouldn't have more auxiliary variables of the same name"
            s1_emb = variables[0]
            logger.info("Adding similarity tracking for " + name)
            # A bit sloppy about downcast

            if "dict" in name:
                embedder = construct_dict_embedder(
                    theano.function([s1, defs, def_mask, s1_def_map], s1_emb, allow_input_downcast=True),
                    vocab=data.vocab, retrieval=retrieval_all)
                extensions.append(
                    SimilarityWordEmbeddingEval(embedder=embedder, prefix=name, every_n_batches=c['mon_freq_valid'],
                        before_training=not fast_start))
            else:
                embedder = construct_embedder(theano.function([s1], s1_emb, allow_input_downcast=True),
                    vocab=data.vocab)
                extensions.append(
                    SimilarityWordEmbeddingEval(embedder=embedder, prefix=name, every_n_batches=c['mon_freq_valid'],
                        before_training=not fast_start))

    extensions.extend([DumpCSVSummaries(
        save_path,
        every_n_batches=c['mon_freq_train'],
        after_training=True),
        Printing(every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches'])])

    logger.info(extensions)

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
        assert np.all(simple.get_embeddings_lookups()[0].parameters[0].get_value(0)[123] == embeddings[123])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=model,
        extensions=extensions)

    assert os.path.exists(save_path)
    main_loop.run()
