"""
Training loop for simple SNLI model that can use dict enchanced embeddings

TODO: What's best way to refactor it. I think training loop taking model + cost would be good enough for us
TODO: State CSV pd.DataFrame(self._current_log).to_csv(os.path.join(self._save_path, "logs.csv"))
"""

import sys

import numpy as np
from theano import tensor

sys.path.append("..")

from blocks.bricks.cost import MisclassificationRate
from blocks.filter import get_brick
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.extensions import ProgressBar, Timestamp
from blocks.extensions.training import TrackTheBest
from dictlearn.extensions import (
    DumpTensorflowSummaries)
from blocks.extensions.predicates import OnLogRecord
from blocks.serialization import load_parameters
from blocks.initialization import Constant, Uniform

from dictlearn.vocab import Vocabulary
from dictlearn.inits import GlorotUniform

import os
import time
import atexit
import signal
import pprint
import pandas as pd
import subprocess
import tqdm
import json
import numpy
import theano
import logging

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
    construct_dict_embedder, RetrievalPrintStats, PrintMessage
from dictlearn.data import SNLIData
from dictlearn.nli_simple_model import NLISimple
from dictlearn.nli_esim_model import ESIM
from dictlearn.retrieval import Retrieval, Dictionary
from dictlearn.nli_simple_model import LSTMReadDefinitions, MeanPoolReadDefinitions, MeanPoolCombiner

from blocks.serialization import secure_dump, dump_and_add_to_dump
from blocks.extensions import SimpleExtension

# vocab defaults to data.vocab
# vocab_text defaults to vocab
# Vocab def defaults to vocab
def _initialize_simple_model_and_data(c):

    if c['vocab']:
        vocab = Vocabulary(c['vocab'])
    else:
        vocab = None
    # Load data
    data = SNLIData(path=c['data_path'], layout=c['layout'], vocab=vocab)

    if vocab is None:
        vocab = data.vocab

    # Dict
    if c['dict_path']:
        dict = Dictionary(c['dict_path'])

        if len(c['vocab_def']):
            retrieval_vocab = Vocabulary(c['vocab_def'])
        else:
            retrieval_vocab = data.vocab

        retrieval = Retrieval(vocab_text=data.vocab, vocab_def=retrieval_vocab,
            dictionary=dict, max_def_length=c['max_def_length'],
            with_too_long_defs=c['with_too_long_defs'],
            exclude_top_k=c['exclude_top_k'], max_def_per_word=c['max_def_per_word'])

        data.set_retrieval(retrieval)
    else:
        retrieval = None
        dict = None
        retrieval_vocab = None

    # Initialize
    simple = NLISimple(
        # Baseline arguments
        emb_dim=c['emb_dim'], vocab=data.vocab, encoder=c['encoder'], dropout=c['dropout'],
        num_input_words=c['num_input_words'], mlp_dim=c['mlp_dim'],

        # Dict lookup kwargs (will get refactored)
        translate_dim=c['translate_dim'], retrieval=retrieval, compose_type=c['compose_type'],
        reader_type=c['reader_type'], disregard_word_embeddings=c['disregard_word_embeddings'],
        def_vocab=retrieval_vocab, def_emb_dim=c['def_emb_dim'],
        combiner_dropout=c['combiner_dropout'], share_def_lookup=c['share_def_lookup'],
        combiner_dropout_type=c['combiner_dropout_type'], combiner_bn=c['combiner_bn'],
        combiner_gating=c['combiner_gating'], combiner_shortcut=c['combiner_shortcut'],
        combiner_reader_translate=c['combiner_reader_translate'], def_dim=c['def_dim'],
        num_input_def_words=c['num_input_def_words'],

        # Init
        weights_init=GlorotUniform(), biases_init=Constant(0.0)
    )
    simple.push_initialization_config()
    if c['encoder'] == 'rnn':
        simple._rnn_encoder.weights_init = Uniform(std=0.1)
    simple.initialize()

    if c['embedding_path']:
        embeddings = np.load(c['embedding_path'])
        simple.set_embeddings(embeddings.astype(theano.config.floatX))

    return simple, data, dict, retrieval, vocab

def _initialize_esim_model_and_data(c):

    if c['vocab']:
        vocab = Vocabulary(c['vocab'])
    else:
        vocab = None

    # Load data
    data = SNLIData(path=c['data_path'], layout=c['layout'], vocab=vocab)

    if vocab is None:
        vocab = data.vocab

    # Dict
    if c['dict_path']:
        dict = Dictionary(c['dict_path'])

        if len(c['vocab_def']):
            retrieval_vocab = Vocabulary(c['vocab_def'])
        else:
            retrieval_vocab = data.vocab

        retrieval = Retrieval(vocab_text=data.vocab, vocab_def=retrieval_vocab,
            dictionary=dict, max_def_length=c['max_def_length'],
            with_too_long_defs=c['with_too_long_defs'],
            exclude_top_k=c['exclude_top_k'], max_def_per_word=c['max_def_per_word'])

        data.set_retrieval(retrieval)

        num_input_def_words = c['num_input_def_words'] if c['num_input_def_words'] > 0 else c['num_input_words']
        def_emb_dim = c['def_emb_dim'] if c['def_emb_dim'] > 0 else c['emb_dim']

        # TODO: Refactor lookup passing to reader. Very incoventient ATM
        if c['reader_type'] == "rnn":
            def_reader = LSTMReadDefinitions(num_input_words=num_input_def_words,
                weights_init=Uniform(width=0.1), translate=c['combiner_reader_translate'],
                biases_init=Constant(0.), dim=c['def_dim'], emb_dim=def_emb_dim,
                vocab=vocab, lookup=None)
        elif c['reader_type'] == "mean":
           def_reader = MeanPoolReadDefinitions(num_input_words=num_input_def_words,
                translate=c['combiner_reader_translate'], vocab=vocab,
                weights_init=Uniform(width=0.1), lookup=None, dim=def_emb_dim,
                biases_init=Constant(0.), emb_dim=def_emb_dim)
        else:
            raise NotImplementedError()

        def_combiner = MeanPoolCombiner(dim=c['def_dim'], emb_dim=def_emb_dim,
            dropout=c['combiner_dropout'], dropout_type=c['combiner_dropout_type'],
            def_word_gating=c['combiner_gating'],
            shortcut_unk_and_excluded=c['combiner_shortcut'], num_input_words=num_input_def_words,
            exclude_top_k=c['exclude_top_k'],
            vocab=vocab, compose_type=c['compose_type'],
            weights_init=Uniform(width=0.1), biases_init=Constant(0.))

    else:
        retrieval = None
        dict = None
        def_combiner = None
        def_reader = None

    # Initialize
    simple = ESIM(
        # Baseline arguments
        emb_dim=c['emb_dim'], vocab=data.vocab, encoder=c['encoder'], dropout=c['dropout'],
        num_input_words=c['num_input_words'], def_dim=c['def_dim'], dim=c['dim'],
        bn=c.get('bn', True),

        def_combiner=def_combiner, def_reader=def_reader,

        # Init
        weights_init=GlorotUniform(), biases_init=Constant(0.0)
    )
    simple.push_initialization_config()
    # TODO: Not sure anymore why we do that
    if c['encoder'] == 'bilstm':
        for enc in simple._rnns:
            enc.weights_init = Uniform(std=0.1)
    simple.initialize()

    if c['embedding_path']:
        embeddings = np.load(c['embedding_path'])
        simple.set_embeddings(embeddings.astype(theano.config.floatX))

    return simple, data, dict, retrieval, vocab

def train_snli_model(new_training_job, config, save_path, params, fast_start, fuel_server, seed, model='simple'):
    if config['exclude_top_k'] > config['num_input_words'] and config['num_input_words'] > 0:
        raise Exception("Some words have neither word nor def embedding")

    c = config
    logger = configure_logger(name="snli_baseline_training", log_file=os.path.join(save_path, "log.txt"))
    if not os.path.exists(save_path):
        logger.info("Start a new job")
        os.mkdir(save_path)
    else:
        logger.info("Continue an existing job")
    with open(os.path.join(save_path, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    main_loop_path = os.path.join(save_path, 'main_loop.tar')
    main_loop_best_val_path = os.path.join(save_path, 'main_loop_best_val.tar')
    stream_path = os.path.join(save_path, 'stream.pkl')

    # Save config to save_path
    json.dump(config, open(os.path.join(save_path, "config.json"), "w"))

    if model == 'simple':
        nli_model, data, used_dict, used_retrieval, _ = _initialize_simple_model_and_data(c)
    elif model == 'esim':
        nli_model, data, used_dict, used_retrieval, _ = _initialize_esim_model_and_data(c)
    else:
        raise NotImplementedError()

    # Compute cost
    s1, s2 = T.lmatrix('sentence1'), T.lmatrix('sentence2')

    if c['dict_path']:
        assert os.path.exists(c['dict_path'])
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
        pred = nli_model.apply(s1, s1_mask, s2, s2_mask, def_mask=def_mask, defs=defs, s1_def_map=s1_def_map,
            s2_def_map=s2_def_map, train_phase=train_phase)
        cost = CategoricalCrossEntropy().apply(y.flatten(), pred)
        error_rate = MisclassificationRate().apply(y.flatten(), pred)
        # NOTE: Please don't change outputs of cg
        cg[train_phase] = ComputationGraph([cost, error_rate])
        if c.get('bn', True):
            cg[train_phase] = apply_batch_normalization(cg[train_phase])

    # Weight decay (TODO: Make it less bug prone)
    if model == 'simple':
        weights_to_decay = VariableFilter(bricks=[dense for dense, relu, bn in nli_model._mlp],
            roles=[WEIGHT])(cg[True].variables)
        weight_decay = np.float32(c['l2']) * sum((w ** 2).sum() for w in weights_to_decay)
    elif model == 'esim':
        weight_decay = 0.0
    else:
        raise NotImplementedError()

    final_cost = cg[True].outputs[0] + weight_decay
    final_cost.name = 'final_cost'

    # Add updates for population parameters

    if c.get("bn", True):
        pop_updates = get_batch_normalization_updates(cg[True])
        extra_updates = [(p, m * 0.1 + p * (1 - 0.1))
            for p, m in pop_updates]
    else:
        pop_updates = []
        extra_updates = []

    if params:
        logger.debug("Load parameters from {}".format(params))
        with open(params) as src:
            loaded_params = load_parameters(src)
            cg[True].set_parameter_values(loaded_params)
            for param, m in pop_updates:
                param.set_value(loaded_params[get_brick(param).get_hierarchical_name(param)])

    if os.path.exists(os.path.join(save_path, "main_loop.tar")):
        logger.warning("Manually loading BN stats :(")
        with open(os.path.join(save_path, "main_loop.tar")) as src:
            loaded_params = load_parameters(src)

        for param, m in pop_updates:
            param.set_value(loaded_params[get_brick(param).get_hierarchical_name(param)])

    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4)
                .get_epoch_iterator())
        s1.tag.test_value = test_value_data[0]
        s1_mask.tag.test_value = test_value_data[1]
        s2.tag.test_value = test_value_data[2]
        s2_mask.tag.test_value = test_value_data[3]
        y.tag.test_value = test_value_data[4]

    # Freeze embeddings
    if not c['train_emb']:
        frozen_params = [p for E in nli_model.get_embeddings_lookups() for p in E.parameters]
        train_params =  [p for p in cg[True].parameters]
        assert len(set(frozen_params) & set(train_params)) > 0
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

    ### Monitored args ###
    train_monitored_vars = [final_cost] + cg[True].outputs
    monitored_vars = cg[False].outputs
    val_acc = monitored_vars[1]
    to_monitor_names = ['def_unk_ratio', 's1_merged_input_rootmean2', 's1_def_mean_rootmean2',
        's1_gate_rootmean2', 's1_compose_gate_rootmean2']
    for k in to_monitor_names:
        train_v, valid_v = VariableFilter(name=k)(cg[True]), VariableFilter(name=k)(cg[False])
        if len(train_v):
            logger.info("Adding {} tracking".format(k))
            train_monitored_vars.append(train_v[0])
            monitored_vars.append(valid_v[0])
        else:
            logger.warning("Didnt find {} in cg".format(k))

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
        seed=seed)

    if fuel_server:
        # the port will be configured by the StartFuelServer extension
        training_stream = ServerDataStream(
            sources=regular_training_stream.sources, hwm=100,
            produces_examples=regular_training_stream.produces_examples)
    else:
        training_stream = regular_training_stream

    ### Build extensions ###

    extensions = [
        Load(main_loop_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job),
        StartFuelServer(regular_training_stream,
            stream_path,
            hwm=100,
            script_path=os.path.join(os.path.dirname(__file__), "../bin/start_fuel_server.py"),
            before_training=fuel_server),
        Timing(every_n_batches=c['mon_freq']),
        ProgressBar(),
        RetrievalPrintStats(retrieval=used_retrieval, every_n_batches=c['mon_freq_valid'],
            before_training=not fast_start),
        Timestamp(),
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq']),
    ]

    if c['layout'] == 'snli':
        validation = DataStreamMonitoring(
            monitored_vars,
            data.get_stream('valid', batch_size=1, seed=seed),
            before_training=not fast_start,
            on_resumption=True,
            after_training=True,
            every_n_batches=c['mon_freq_valid'],
            prefix='valid')
        extensions.append(validation)
    elif c['layout'] == 'mnli':
        validation = DataStreamMonitoring(
            monitored_vars,
            data.get_stream('valid_matched', batch_size=1, seed=seed),
            every_n_batches=c['mon_freq_valid'],
            on_resumption=True,
            after_training=True,
            prefix='valid_matched')
        validation_mismatched = DataStreamMonitoring(
            monitored_vars,
            data.get_stream('valid_mismatched', batch_size=1, seed=seed),
            every_n_batches=c['mon_freq_valid'],
            before_training=not fast_start,
            on_resumption=True,
            after_training=True,
            prefix='valid_mismatched')
        extensions.extend([validation, validation_mismatched])
    else:
        raise NotImplementedError()

    # Similarity trackers for embeddings
    if len(c['vocab_def']):
        retrieval_vocab = Vocabulary(c['vocab_def'])
    else:
        retrieval_vocab = data.vocab

    retrieval_all = Retrieval(vocab_text=retrieval_vocab, dictionary=used_dict, max_def_length=c['max_def_length'],
        exclude_top_k=0, max_def_per_word=c['max_def_per_word'])

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

    track_the_best = TrackTheBest(
        validation.record_name(val_acc),
        before_training=not fast_start,
        every_n_epochs=c['save_freq_epochs'],
        after_training=not fast_start,
        every_n_batches=c['mon_freq_valid'],
        choose_best=min)
    extensions.append(track_the_best)
    extensions.append(Checkpoint(main_loop_path,
        parameters=cg[True].parameters + [p for p, m in pop_updates],
        before_training=not fast_start,
        every_n_epochs=c['save_freq_epochs'],
        after_training=not fast_start).add_condition(
        ['after_batch', 'after_epoch'],
        OnLogRecord(track_the_best.notification_name),
        (main_loop_best_val_path,)))
    extensions.extend([DumpCSVSummaries(
        save_path,
        every_n_batches=c['mon_freq'],
        after_training=True),
        DumpTensorflowSummaries(
            save_path,
            after_epoch=True,
            every_n_batches=c['mon_freq'],
            after_training=True),
        Printing(every_n_batches=c['mon_freq']),
        PrintMessage(msg="save_path={}".format(save_path), every_n_batches=c['mon_freq']),
        FinishAfter(after_n_batches=c['n_batches'])])
    logger.info(extensions)

    ### Run training ###

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

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=model,
        extensions=extensions)

    assert os.path.exists(save_path)
    main_loop.run()

def evaluate(c, tar_path, *args, **kwargs):
    """
    * Runs on valid and test given network
    * Saves all predictions
    * Saves results.json and predictions.csv
    """

    # Load and configure
    model = kwargs['model']
    assert c.endswith("json")
    c = json.load(open(c))
    # Very ugly absolute path fix
    ABS_PATH = "/mnt/users/jastrzebski/local/dict_based_learning/"
    from six import string_types
    for k in c:
        if isinstance(c[k], string_types):
            if c[k].startswith(ABS_PATH):
                c[k] = c[k][len(ABS_PATH):]
    assert tar_path.endswith("tar")
    dest_path = os.path.dirname(tar_path)
    prefix = os.path.splitext(os.path.basename(tar_path))[0]

    s1_decoded, s2_decoded = T.lmatrix('sentence1'), T.lmatrix('sentence2')

    if c['dict_path']:
        s1_def_map, s2_def_map = T.lmatrix('sentence1_def_map'), T.lmatrix('sentence2_def_map')
        def_mask = T.fmatrix("def_mask")
        defs = T.lmatrix("defs")
    else:
        s1_def_map, s2_def_map = None, None
        def_mask = None
        defs = None

    s1_mask, s2_mask = T.fmatrix('sentence1_mask'), T.fmatrix('sentence2_mask')

    if model == 'simple':
        model, data, used_dict, used_retrieval, used_vocab = _initialize_simple_model_and_data(c)
    elif model == 'esim':
        model, data, used_dict, used_retrieval, used_vocab = _initialize_esim_model_and_data(c)
    else:
        raise NotImplementedError()

    pred = model.apply(s1_decoded, s1_mask, s2_decoded, s2_mask, def_mask=def_mask, defs=defs, s1_def_map=s1_def_map,
        s2_def_map=s2_def_map, train_phase=False)

    cg = ComputationGraph([pred])
    cg = apply_batch_normalization(cg)

    if c.get("bn", True):
        bn_params = [p for p, update_p in get_batch_normalization_updates(cg)]
    else:
        bn_params = []

    # cg = ComputationGraph([pred])

    # Load model
    model = Model(cg.outputs)
    parameters = model.get_parameter_dict()  # Blocks version mismatch
    logging.info("Trainable parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                        for key in sorted([get_brick(param).get_hierarchical_name(param) for param in cg.parameters])],
                    width=120))
    logging.info("# of parameters {}".format(
        sum([np.prod(parameters[key].get_value().shape)
            for key in sorted([get_brick(param).get_hierarchical_name(param) for param in cg.parameters])])))
    with open(tar_path) as src:
        params = load_parameters(src)

        loaded_params_set = set(params.keys())
        model_params_set = set([get_brick(param).get_hierarchical_name(param) for param in cg.parameters])

        logging.info("Loaded extra parameters")
        logging.info(loaded_params_set - model_params_set)
        logging.info("Missing parameters")
        logging.info(model_params_set - loaded_params_set)
    model.set_parameter_values(params)

    if c.get("bn", True):
        logging.info("Loading " + str([get_brick(param).get_hierarchical_name(param) for param in bn_params]))
        for param in bn_params:
            param.set_value(params[get_brick(param).get_hierarchical_name(param)])
        for p in bn_params:
            model._parameter_dict[get_brick(p).get_hierarchical_name(p)] = p

    # Read logs
    logs = pd.read_csv(os.path.join(dest_path, "logs.csv"))
    best_val_acc = logs['valid_misclassificationrate_apply_error_rate'].min()
    logging.info("Best measured valid acc: " + str(best_val_acc))

    # Predict
    predict_fnc = theano.function(cg.inputs, pred)

    results = {}

    # TODO: Depends on batch_size?
    batch_size = 14
    for subset in ['valid', 'test']:
        logging.info("Predicting on " + subset)
        stream = data.get_stream(subset, batch_size=batch_size, seed=778)
        it = stream.get_epoch_iterator()
        rows = []
        for ex in tqdm.tqdm(it, total=10000/batch_size):
            ex = dict(zip(stream.sources, ex))
            inp = [ex[v.name] for v in cg.inputs]
            prob = predict_fnc(*inp)
            label_pred = np.argmax(prob, axis=1)

            for id in range(len(prob)):
                s1_decoded = used_vocab.decode(ex['sentence1'][id]).split()
                s2_decoded = used_vocab.decode(ex['sentence2'][id]).split()

                assert used_vocab == data.vocab

                s1_decoded = ['*' + w + '*' if used_vocab.word_to_id(w) > c['num_input_words'] else w for w in s1_decoded]
                s2_decoded = ['*' + w + '*' if used_vocab.word_to_id(w) > c['num_input_words'] else w for w in s2_decoded]

                # Different difficulty metrics

                # text_unk_percentage
                s1_no_pad = [w for w in ex['sentence1'][id] if w != 0]
                s2_no_pad = [w for w in ex['sentence2'][id] if w != 0]

                s1_unk_percentage = sum([1. for w in s1_no_pad if w == used_vocab.unk]) / len(s1_no_pad)
                s2_unk_percentage = sum([1. for w in s1_no_pad if w == used_vocab.unk]) / len(s2_no_pad)

                # mean freq word
                s1_mean_freq = np.mean([0 if w == data.vocab.unk else used_vocab._id_to_freq[w] for w in s1_no_pad])
                s2_mean_freq = np.mean([0 if w == data.vocab.unk else used_vocab._id_to_freq[w] for w in s2_no_pad])

                rows.append({"pred": label_pred[id], "true_label": ex['label'][id],
                    "s1": ' '.join(s1_decoded),
                    "s2": ' '.join(s2_decoded),
                    "s1_unk_percentage": s1_unk_percentage,
                    "s2_unk_percentage": s2_unk_percentage,
                    "s1_mean_freq": s1_mean_freq,
                    "s2_mean_freq": s2_mean_freq,
                    "p_0": prob[id, 0], "p_1": prob[id, 1], "p_2": prob[id, 2]})

        preds = pd.DataFrame(rows, columns=rows[0].keys())
        preds.to_csv(os.path.join(dest_path, prefix + '_predictions_{}.csv'.format(subset)))
        results[subset] = {}
        results[subset]['misclassification'] = 1 - np.mean(preds.pred == preds.true_label)

        if subset == "valid" and np.abs(( 1 - np.mean(preds.pred == preds.true_label)) - best_val_acc) > 0.001:
            logging.error("!!!")
            logging.error("Found different best_val_acc. Probably due to changed specification of the model class.")
            # TODO: Why this discrepancy? It is around 0.5% usually
            logging.error("Discrepancy {}".format(( 1 - np.mean(preds.pred == preds.true_label)) - best_val_acc))
            logging.error("!!!")

        logging.info(results)

        # Save useful plots

    json.dump(results, open(os.path.join(dest_path, prefix + '_results.json'), "w"))


