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
import json

import numpy
import theano
from theano import tensor
from nltk.tokenize.moses import MosesDetokenizer

from blocks.initialization import Uniform, Constant
from blocks.bricks.recurrent import Bidirectional
from blocks.bricks.simple import Rectifier
from blocks.algorithms import (
    Adam, GradientDescent, Adam, StepClipping, CompositeRule)
from blocks.graph import ComputationGraph, apply_dropout
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.serialization import load_parameters
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.extensions.predicates import OnLogRecord

import fuel
from fuel.streams import ServerDataStream

from dictlearn.util import (
    rename, masked_root_mean_square, get_free_port,
    copy_streams_to_file, run_with_redirection)
from dictlearn.theano_util import parameter_stats, unk_ratio
from dictlearn.data import ExtractiveQAData
from dictlearn.extensions import (
    DumpTensorflowSummaries, LoadNoUnpickling, StartFuelServer,
    RetrievalPrintStats)
from dictlearn.extractive_qa_model import ExtractiveQAModel
from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Retrieval, Dictionary

logger = logging.getLogger()


def _initialize_data_and_model(config):
    c = config
    vocab = None
    if c['vocab_path']:
        vocab = Vocabulary(
            os.path.join(fuel.config.data_path[0], c['vocab_path']))
    data = ExtractiveQAData(path=c['data_path'], vocab=vocab, layout=c['layout'])
    # TODO: fix me, I'm so ugly
    if c['dict_path']:
        dict_vocab = data.vocab
        if c['dict_vocab_path']:
            dict_vocab = Vocabulary(
                os.path.join(fuel.config.data_path[0], c['dict_vocab_path']))
        data._retrieval = Retrieval(
            dict_vocab, Dictionary(
                os.path.join(fuel.config.data_path[0], c['dict_path'])),
            c['max_def_length'], c['exclude_top_k'])
    qam = ExtractiveQAModel(
        c['dim'], c['emb_dim'], c['readout_dims'],
        c['coattention'], c['num_input_words'],
        data.vocab,
        use_definitions=bool(c['dict_path']),
        def_word_gating=c['def_word_gating'],
        compose_type=c['compose_type'],
        reuse_word_embeddings=c['reuse_word_embeddings'],
        def_reader=c['def_reader'],
        weights_init=Uniform(width=0.1),
        biases_init=Constant(0.))
    qam.initialize()
    logger.debug("Model created")
    if c['embedding_path']:
        qam.set_embeddings(numpy.load(
            os.path.join(fuel.config.data_path[0], c['embedding_path'])))
    logger.debug("Embeddings loaded")
    return data, qam


def train_extractive_qa(new_training_job, config, save_path,
                              params, fast_start, fuel_server):
    root_path = os.path.join(save_path, 'training_state')
    extension = '.tar'
    tar_path = root_path + extension

    c = config
    data, qam = _initialize_data_and_model(c)

    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', shuffle=True, batch_size=4, max_length=5)
            .get_epoch_iterator(as_dict=True))
        for var in qam.input_vars.values():
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
    context_unk_ratio, = VariableFilter(name='context_unk_ratio')(cg)
    monitored_vars = [length, batch_size, cost, exact_match_ratio,
                      context_unk_ratio]
    if c['dict_path']:
        def_unk_ratio, = VariableFilter(name='def_unk_ratio')(cg)
        num_definitions = rename(qam.input_vars['defs'].shape[0],
                                 'num_definitions')
        max_definition_length = rename(qam.input_vars['defs'].shape[1],
                                       'max_definition_length')
        monitored_vars.extend([def_unk_ratio, num_definitions, max_definition_length])
        if c['def_word_gating'] == 'self_attention':
            def_gates = VariableFilter(name='def_gates')(cg)
            def_gates_min = tensor.minimum(*[x.min() for x in def_gates])
            def_gates_max = tensor.maximum(*[x.max() for x in def_gates])
            monitored_vars.extend([rename(def_gates_min, 'def_gates_min'),
                                   rename(def_gates_max, 'def_gates_max')])

    parameters = cg.get_parameter_dict()
    trained_parameters = parameters.values()
    if c['embedding_path']:
        logger.debug("Exclude  word embeddings from the trained parameters")
        trained_parameters = [p for p in trained_parameters
                              if not p == qam.embeddings_var()]
    logger.info("Cost parameters" + "\n" +
                pprint.pformat(
                    [" ".join((
                       key, str(parameters[key].get_value().shape),
                       'trained' if parameters[key] in trained_parameters else 'frozen'))
                     for key in sorted(parameters.keys())],
                    width=120))

    # apply dropout to the training cost and to all the variables
    # that we monitor during training
    if c['dropout']:
        train_monitored_vars = list(monitored_vars)
        regularized_cg = ComputationGraph([cost] + train_monitored_vars)
        bidir_outputs, = VariableFilter(
            bricks=[Bidirectional], roles=[OUTPUT])(cg)
        readout_layers = VariableFilter(
            bricks=[Rectifier], roles=[OUTPUT])(cg)
        dropout_vars = [bidir_outputs] + readout_layers
        logger.debug("applying dropout to {}".format(
            ", ".join([v.name for v in  dropout_vars])))
        regularized_cg = apply_dropout(regularized_cg, dropout_vars, c['dropout'])
        train_cost = regularized_cg.outputs[0]
        train_monitored_vars = regularized_cg.outputs[1:]

    rules = []
    if c['grad_clip_threshold']:
        rules.append(StepClipping(c['grad_clip_threshold']))
    rules.append(Adam(learning_rate=c['learning_rate'],
                      beta1=c['momentum']))
    algorithm = GradientDescent(
        cost=train_cost,
        parameters=trained_parameters,
        step_rule=CompositeRule(rules))

    if c['grad_clip_threshold']:
        train_monitored_vars.append(algorithm.total_gradient_norm)
    if c['monitor_parameters']:
        train_monitored_vars.extend(parameter_stats(parameters, algorithm))

    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'],
        shuffle=True, max_length=c['max_length'])
    original_training_stream = training_stream
    if fuel_server:
        # the port will be configured by the StartFuelServer extension
        training_stream = ServerDataStream(
            sources=training_stream.sources,
            produces_examples=training_stream.produces_examples)

    extensions = [
        LoadNoUnpickling(tar_path, load_iteration_state=True, load_log=True)
           .set_conditions(before_training=not new_training_job),
        StartFuelServer(original_training_stream,
                        os.path.join(save_path, 'stream.pkl'),
                        before_training=fuel_server),
        Timing(every_n_batches=c['mon_freq_train']),
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
    ]
    validation = DataStreamMonitoring(
        monitored_vars,
        data.get_stream('dev', batch_size=c['batch_size_valid']),
        prefix="dev").set_conditions(
            before_training=not fast_start,
            after_epoch=True)
    track_the_best = TrackTheBest(
        validation.record_name(exact_match_ratio),
        choose_best=max).set_conditions(
            before_training=True,
            after_epoch=True)
    extensions.extend([validation,
                       track_the_best])
        # We often use pretrained word embeddings and we don't want
        # to load and save them every time. To avoid that, we use
        # save_main_loop=False, we only save the trained parameters,
        # and we save the log and the iterations state separately
        # in the tar file.
    extensions.extend([
        Checkpoint(tar_path,
                   parameters=trained_parameters,
                   save_main_loop=False,
                   save_separately=['log', 'iteration_state'],
                   before_training=not fast_start,
                   every_n_epochs=c['save_freq_epochs'],
                   every_n_batches=c['save_freq_batches'],
                   after_training=not fast_start)
            .add_condition(
                ['after_batch', 'after_epoch'],
                 OnLogRecord(track_the_best.notification_name),
                 (root_path + "_best" + extension,)),
        DumpTensorflowSummaries(
            save_path,
            after_epoch=True,
            every_n_batches=c['mon_freq_train'],
            after_training=True),
        #RetrievalPrintStats(
        #    retrieval=data._retrieval, every_n_batches=c['mon_freq_train'],
        #    before_training=not fast_start),
        Printing(after_epoch=True,
                 every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches'])
    ])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()


def evaluate_extractive_qa(config, tar_path, part, num_examples, dest_path):
    c = config
    data, qam = _initialize_data_and_model(c)
    costs = qam.apply_with_default_vars()
    cg = Model(costs)

    with open(tar_path) as src:
        cg.set_parameter_values(load_parameters(src))

    detok = MosesDetokenizer()
    def detokenize(str_):
        return " ".join(detok.detokenize(str_))

    predicted_begins, = VariableFilter(name='predicted_begins')(cg)
    predicted_ends, = VariableFilter(name='predicted_ends')(cg)
    compute = [predicted_begins[0], predicted_ends[0]]
    if c['coattention']:
        d2q_att_weights, = VariableFilter(name='d2q_att_weights')(cg)
        q2d_att_weights, = VariableFilter(name='q2d_att_weights')(cg)
        compute.extend([d2q_att_weights, q2d_att_weights])
    predict_func = theano.function(qam.input_vars, compute)

    done_examples = 0
    num_correct = 0
    def print_stats():
        print('EXACT MATCH RATIO: {}'.format(num_correct / float(done_examples)))

    d2q = []
    q2d = []
    predictions = {}

    stream = data.get_stream(part, batch_size=1, shuffle=part == 'train',
                             raw_text=True, q_ids=True)
    for example in stream.get_epoch_iterator(as_dict=True):
        if done_examples == num_examples:
            break
        feed = dict(example)
        del feed['q_ids']
        feed['contexts'] = numpy.array(data.vocab.encode(example['contexts'][0]))[None, :]
        feed['questions'] = numpy.array(data.vocab.encode(example['questions'][0]))[None, :]
        result = predict_func(**feed)
        correct_answer_span = slice(example['answer_begins'], example['answer_ends'])
        predicted_answer_span = slice(*result[:2])
        is_correct = correct_answer_span == predicted_answer_span
        answer = detokenize(example['contexts'][0, predicted_answer_span])

        if c['coattention']:
            d2q.append(result[-2])
            q2d.append(result[-1])

        done_examples += 1
        num_correct += is_correct
        predictions[example['q_ids'][0]] = answer

        result = 'correct' if is_correct else 'wrong'
        print('#{}'.format(done_examples))
        print("CONTEXT:", detokenize(example['contexts'][0]))
        print("QUESTION:", detokenize(example['questions'][0]))
        print("ANSWER (span=[{}, {}], {}):".format(predicted_answer_span.start,
                                                   predicted_answer_span.stop,
                                                   result),
              answer)
        print()

        if done_examples % 100 == 0:
            print_stats()
    print_stats()

    # with open(os.path.join(save_path, 'attention.pkl'), 'w') as dst:
    #     cPickle.dump({'d2q': d2q, 'q2d': q2d}, dst)
    with open(dest_path, 'w') as dst:
        json.dump(predictions, dst, indent=2)
