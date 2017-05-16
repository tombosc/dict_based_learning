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

import blocks
from blocks.initialization import Uniform, Constant
from blocks.algorithms import (
    Adam, GradientDescent, Adam, StepClipping, CompositeRule)
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Load, Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord

from blocks.main_loop import MainLoop
from blocks.serialization import load_parameters

import fuel
from fuel.streams import ServerDataStream

from dictlearn.util import rename, masked_root_mean_square, get_free_port
from dictlearn.theano_util import parameter_stats
from dictlearn.data import LanguageModellingData
from dictlearn.extensions import (
    DumpTensorflowSummaries, StartFuelServer, LoadNoUnpickling,
    RetrievalPrintStats)

from dictlearn.language_model import LanguageModel
from dictlearn.retrieval import Retrieval, Dictionary
from dictlearn.vocab import Vocabulary

from tests.util import temporary_content_path

logger = logging.getLogger()

def initialize_data_and_model(config):
    c = config
    fuel_path = fuel.config.data_path[0]
    vocab=None
    if c['vocab_path']:
        vocab = Vocabulary(
            os.path.join(fuel.config.data_path[0], c['vocab_path']))

    data = LanguageModellingData(c['data_path'], c['layout'], vocab=vocab)
    retrieval = None
    if c['dict_path'] and not c['embedding_path']:
        dict_full_path = os.path.join(fuel_path, c['dict_path'])
        dict_ = Dictionary(dict_full_path)
        logger.debug("Loaded dictionary with {} entries"
                     .format(dict_.num_entries()))
        vocab_dict = data.vocab
        if c['dict_vocab_path']:
            if not c['standalone_def_lookup']:
                raise ValueError("Standalone def lookup mandatory with separate vocabs")
            vocab_dict = Vocabulary(
                os.path.join(fuel.config.data_path[0], c['dict_vocab_path']))
                
        retrieval = Retrieval(vocab_dict, dict_,
                              c['max_def_length'], c['exclude_top_k'],
                              max_def_per_word=c['max_def_per_word'])
    elif c['embedding_path']:
        assert(c['dict_path'])
        emb_full_path = os.path.join(fuel_path, c['embedding_path'])
        embedding_matrix = numpy.load(emb_full_path)
        dict_full_path = os.path.join(fuel_path, c['dict_path'])
        dict_ = Dictionary(dict_full_path) # should be key=value=word
        if not c['standalone_def_lookup']:
            raise ValueError("Standalone def lookup mandatory")

        retrieval = Retrieval(data.vocab, dict_, max_def_length=1, 
                              exclude_top_k=c['exclude_top_k'],
                              max_def_per_word=1, add_bod_eod=False)

    lm = LanguageModel(c['emb_dim'], c['emb_def_dim'], c['dim'], c['num_input_words'],
                       c['def_num_input_words'], c['num_output_words'], data.vocab,
                       retrieval,
                       c['def_reader'],
                       c['standalone_def_lookup'],
                       c['standalone_def_rnn'],
                       c['disregard_word_embeddings'],
                       c['compose_type'],
                       weights_init=Uniform(width=0.1),
                       biases_init=Constant(0.))
    lm.initialize()
    
    if c['embedding_path']:
        lm.set_def_embeddings(embedding_matrix)
        logger.debug("Embeddings loaded")

    return (data, lm, retrieval)


def train_language_model(new_training_job, config, save_path, params,
                         fast_start, fuel_server, seed):
    c = config
    if seed:
        fuel.config.default_seed = seed
        blocks.config.config.default_seed = seed

    data, lm, retrieval = initialize_data_and_model(config)

    # full main loop can be saved...
    main_loop_path = os.path.join(save_path, 'main_loop.tar')
    # or only state (log + params) which can be useful not to pickle embeddings
    state_path = os.path.join(save_path, 'training_state.tar')
    stream_path = os.path.join(save_path, 'stream.pkl')
    best_tar_path = os.path.join(save_path, "best_model.tar")

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
    perplexity, = VariableFilter(name='perplexity')(cg)
    perplexities = VariableFilter(name_regex='perplexity.*')(cg)
    monitored_vars = [length, cost] + perplexities
    if c['dict_path']:
        num_definitions, = VariableFilter(name='num_definitions')(cg)
        monitored_vars.extend([num_definitions])

    parameters = cg.get_parameter_dict()
    trained_parameters = parameters.values()
    if c['embedding_path']:
        logger.debug("Exclude word embeddings from the trained parameters")
        trained_parameters = [p for p in trained_parameters
                              if not p == lm.get_def_embeddings_params()]

    logger.info("Cost parameters" + "\n" +
                pprint.pformat(
                    [" ".join((
                       key, str(parameters[key].get_value().shape),
                       'trained' if parameters[key] in trained_parameters else 'frozen'))
                     for key in sorted(parameters.keys())],
                    width=120))

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

    word_emb_RMS, = VariableFilter(name='word_emb_RMS')(cg)
    main_rnn_in_RMS, = VariableFilter(name='main_rnn_in_RMS')(cg)
    train_monitored_vars.extend([word_emb_RMS, main_rnn_in_RMS])

    if c['monitor_parameters']:
        train_monitored_vars.extend(parameter_stats(parameters, algorithm))

    # We use a completely random seed on purpose. With Fuel server
    # it's currently not possible to restore the state of the training
    # stream. That's why it's probably better to just have it stateless.
    stream_seed = numpy.random.randint(0, 10000000) if fuel_server else None
    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'], max_length=c['max_length'],
        seed=stream_seed)
    valid_stream = data.get_stream('valid', batch_size=c['batch_size_valid'],
                                max_length=c['max_length'], seed=stream_seed)
    original_training_stream = training_stream
    if fuel_server:
        # the port will be configured by the StartFuelServer extension
        training_stream = ServerDataStream(
            sources=training_stream.sources,
            produces_examples=training_stream.produces_examples)

    validation = DataStreamMonitoring(
        monitored_vars,
        valid_stream,
        prefix="valid").set_conditions(
            before_first_epoch=not fast_start,
            on_resumption = True,
            every_n_batches=c['mon_freq_valid'])
    track_the_best = TrackTheBest(
            validation.record_name(perplexity),
            choose_best=min).set_conditions(
            on_resumption = True,
            after_epoch=True,
            every_n_batches=c['mon_freq_valid'])
   

    # don't save them the entire main loop to avoid pickling everything
    if c['fast_checkpoint']:
        load = (LoadNoUnpickling(state_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job))
        checkpoint = Checkpoint(state_path,
                                parameters=trained_parameters,
                                save_main_loop=False,
                                save_separately=['log','iteration_state'],
                                before_training=not fast_start,
                                every_n_batches=c['save_freq_batches'],
                                after_training=not fast_start)
    else:
        load = (Load(main_loop_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job))
        checkpoint = Checkpoint(main_loop_path,
                                save_separately=['iteration_state'],
                                before_training=not fast_start,
                                every_n_batches=c['save_freq_batches'],
                                after_training=not fast_start)

    checkpoint = checkpoint.add_condition(
                                ['after_batch', 'after_epoch'],
                                OnLogRecord(track_the_best.notification_name),
                                (best_tar_path,))

    extensions = [
            load,
            StartFuelServer(original_training_stream,
                            stream_path,
                            before_training=fuel_server),
            Timing(every_n_batches=c['mon_freq_train'])
        ]

    if retrieval: 
        extensions.append(
            RetrievalPrintStats(retrieval=retrieval,
                                every_n_batches=c['mon_freq_valid'],
                                before_training=not fast_start))

    extensions.extend([
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
        validation,
        track_the_best,
        checkpoint,
        DumpTensorflowSummaries(
            save_path,
            every_n_batches=c['mon_freq_train'],
            after_training=True),
        Printing(on_resumption=True,
                 every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches'])
    ])

    logger.info("monitored variables during training:" + "\n" +
                pprint.pformat(train_monitored_vars, width=120))
    logger.info("monitored variables during valid:" + "\n" +
                pprint.pformat(monitored_vars, width=120))

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)

    main_loop.run()
