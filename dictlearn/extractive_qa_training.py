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

import blocks
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
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.extensions import SimpleExtension
from blocks.extensions.predicates import OnLogRecord

import fuel
from fuel.streams import ServerDataStream

import dictlearn.squad_evaluate
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
from dictlearn.squad_evaluate import normalize_answer
from dictlearn.util import vec2str

logger = logging.getLogger()

detok = MosesDetokenizer()
def detokenize(str_):
    return u" ".join(detok.detokenize(str_))


class TextMatchRatio(MonitoredQuantity):

    def __init__(self, data_path, *args, **kwargs):
        with open(data_path, 'r'):
            self._data = json.load(open(data_path))['data']
        super(TextMatchRatio, self).__init__(*args, **kwargs)

    def initialize(self):
        self.predictions = {}

    def aggregate(self, predicted_begins, predicted_ends,
                  contexts_text, q_ids):
        batch_size = predicted_begins.shape[0]
        for i in range(batch_size):
            predicted_answer = detokenize(
                map(vec2str,
                    contexts_text[i][predicted_begins[i]:predicted_ends[i]]))
            q_id = vec2str(q_ids[i])
            self.predictions[q_id] = predicted_answer

    def get_aggregated_value(self):
        results = dictlearn.squad_evaluate.evaluate(self._data, self.predictions)
        return results['exact_match']


class DumpPredictions(SimpleExtension):

    def __init__(self, save_path, text_match_ratio, *args, **kwargs):
        self._save_path = save_path
        self._text_match_ratio = text_match_ratio
        super(DumpPredictions, self).__init__(*args, **kwargs)

    def do(self, *args, **kwargs):
        iterations_done = self.main_loop.log.status['iterations_done']
        with open(os.path.join(
                self._save_path, "{}.json".format(iterations_done)), 'w') as dst:
            json.dump(self._text_match_ratio.predictions, dst, indent=2, sort_keys=True)


class Annealing(SimpleExtension):

    def __init__(self, annealing_learning_rate, *args, **kwargs):
        self._annealing_learning_rate = annealing_learning_rate
        kwargs['before_training'] = True
        super(Annealing, self).__init__(*args, **kwargs)

    def do(self, which_callback, *args, **kwargs):
        if which_callback == 'before_training':
            cg = ComputationGraph(self.main_loop.algorithm.total_step_norm)
            self._learning_rate_var, = VariableFilter(theano_name='learning_rate')(cg)
            logger.debug("Annealing extension is initialized")
        elif which_callback == 'after_epoch':
            logger.debug("Annealing the learning rate to {}".format(self._annealing_learning_rate))
            self._learning_rate_var.set_value(self._annealing_learning_rate)
        else:
            raise ValueError("don't know what to do")


def initialize_data_and_model(config):
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
            c['max_def_length'], c['exclude_top_k'],
            with_too_long_defs=c['with_too_long_defs'])
    logger.debug("Data loaded")
    qam = ExtractiveQAModel(
        c['dim'], c['emb_dim'], c['readout_dims'],
        c['num_input_words'], c['def_num_input_words'], data.vocab,
        coattention=c['coattention'],
        use_definitions=bool(c['dict_path']),
        def_word_gating=c['def_word_gating'],
        compose_type=c['compose_type'],
        reuse_word_embeddings=c['reuse_word_embeddings'],
        random_unk=c['random_unk'],
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
                        params, fast_start, fuel_server, seed):
    if seed:
        fuel.config.default_seed = seed
        blocks.config.config.default_seed = seed

    root_path = os.path.join(save_path, 'training_state')
    extension = '.tar'
    tar_path = root_path + extension
    best_tar_path = root_path + '_best' + extension

    c = config
    data, qam = initialize_data_and_model(c)

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
    predicted_begins, = VariableFilter(name='predicted_begins')(cg)
    predicted_ends, = VariableFilter(name='predicted_ends')(cg)
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
    text_match_ratio = TextMatchRatio(
        data_path=os.path.join(fuel.config.data_path[0], 'squad/dev-v1.1.json'),
        requires=[predicted_begins, predicted_ends,
                  tensor.ltensor3('contexts_text'),
                  tensor.lmatrix('q_ids')],
        name='text_match_ratio')

    parameters = cg.get_parameter_dict()
    trained_parameters = parameters.values()
    if c['embedding_path']:
        logger.debug("Exclude  word embeddings from the trained parameters")
        trained_parameters = [p for p in trained_parameters
                              if not p == qam.embeddings_var()]
    if c['train_only_def_part']:
        def_reading_parameters = qam.def_reading_parameters()
        trained_parameters = [p for p in trained_parameters
                              if p in def_reading_parameters]

    logger.info("Cost parameters" + "\n" +
                pprint.pformat(
                    [" ".join((
                       key, str(parameters[key].get_value().shape),
                       'trained' if parameters[key] in trained_parameters else 'frozen'))
                     for key in sorted(parameters.keys())],
                    width=120))

    # apply dropout to the training cost and to all the variables
    # that we monitor during training
    train_cost = cost
    train_monitored_vars = list(monitored_vars)
    if c['dropout']:
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
        [text_match_ratio] + monitored_vars,
        data.get_stream('dev', batch_size=c['batch_size_valid'],
                        raw_text=True, q_ids=True),
        prefix="dev").set_conditions(
            before_training=not fast_start,
            after_epoch=True)
    dump_predictions = DumpPredictions(
        save_path, text_match_ratio,
        before_training=not fast_start,
        after_epoch=True)
    track_the_best_exact = TrackTheBest(
        validation.record_name(exact_match_ratio),
        choose_best=max).set_conditions(
            before_training=True,
            after_epoch=True)
    track_the_best_text = TrackTheBest(
        validation.record_name(text_match_ratio),
        choose_best=max).set_conditions(
            before_training=True,
            after_epoch=True)
    extensions.extend([validation, dump_predictions,
                       track_the_best_exact,
                       track_the_best_text])
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
                 OnLogRecord(track_the_best_text.notification_name),
                 (best_tar_path,)),
        DumpTensorflowSummaries(
            save_path,
            after_epoch=True,
            every_n_batches=c['mon_freq_train'],
            after_training=True),
        RetrievalPrintStats(
            retrieval=data._retrieval, every_n_batches=c['mon_freq_train'],
            before_training=not fast_start),
        Printing(after_epoch=True,
                 every_n_batches=c['mon_freq_train']),
        FinishAfter(after_n_batches=c['n_batches']),
        Annealing(c['annealing_learning_rate'],
                  after_n_epochs=c['annealing_start_epoch']),
        LoadNoUnpickling(best_tar_path,
                         after_n_epochs=c['annealing_start_epoch'])
    ])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()


def evaluate_extractive_qa(config, tar_path, part, num_examples, dest_path, qids=None):
    if not dest_path:
        dest_path = os.path.join(os.path.dirname(tar_path), 'predictions.json')
    log_path = os.path.splitext(dest_path)[0] + '_log.json'

    if qids:
        qids = qids.split(',')

    c = config
    data, qam = initialize_data_and_model(c)
    costs = qam.apply_with_default_vars()
    cg = Model(costs)

    with open(tar_path) as src:
        cg.set_parameter_values(load_parameters(src))

    predicted_begins, = VariableFilter(name='predicted_begins')(cg)
    predicted_ends, = VariableFilter(name='predicted_ends')(cg)
    compute = {'begins': predicted_begins, 'ends': predicted_ends}
    if c['coattention']:
        d2q_att_weights, = VariableFilter(name='d2q_att_weights')(cg)
        q2d_att_weights, = VariableFilter(name='q2d_att_weights')(cg)
        compute.update({'d2q': d2q_att_weights,
                        'q2d': q2d_att_weights})
    compute['costs'] = costs
    predict_func = theano.function(qam.input_vars.values(), compute)
    logger.debug("Ready to evaluate")

    done_examples = 0
    num_correct = 0
    def print_stats():
        print('EXACT MATCH RATIO: {}'.format(num_correct / float(done_examples)))

    predictions = {}
    log = {}

    stream = data.get_stream(part, batch_size=1, shuffle=part == 'train',
                             raw_text=True, q_ids=True)
    for example in stream.get_epoch_iterator(as_dict=True):
        if done_examples == num_examples:
            break
        q_id = vec2str(example['q_ids'][0])
        if qids and not q_id in qids:
            continue

        example['contexts_text'] = [
            map(vec2str, example['contexts_text'][0])]
        example['questions_text'] = [
            map(vec2str, example['questions_text'][0])]
        feed = dict(example)
        del feed['q_ids']
        del feed['contexts_text']
        del feed['questions_text']
        del feed['contexts_text_mask']
        result = predict_func(**feed)
        correct_answer_span = slice(example['answer_begins'][0], example['answer_ends'][0])
        predicted_answer_span = slice(result['begins'][0], result['ends'][0])
        correct_answer = example['contexts_text'][0][correct_answer_span]
        answer = example['contexts_text'][0][predicted_answer_span]
        is_correct = correct_answer_span == predicted_answer_span
        context = example['contexts_text'][0]
        question = example['questions_text'][0]

        # pretty print
        outcome = 'correct' if is_correct else 'wrong'
        print('#{}'.format(done_examples))
        print(u"CONTEXT:", detokenize(context))
        print(u"QUESTION:", detokenize(question))
        print(u"RIGHT ANSWER: {}".format(detokenize(correct_answer)))
        print(u"ANSWER (span=[{}, {}], {}):".format(predicted_answer_span.start,
                                                    predicted_answer_span.stop,
                                                    outcome),
              detokenize(answer))
        print()

        # update statistics
        done_examples += 1
        num_correct += is_correct


        # save the results
        predictions[q_id] = detokenize(answer)
        log_entry = {'context': context,
                     'question': question,
                     'answer': answer,
                     'correct_answer': correct_answer,
                     'cost' : float(result['costs'][0])}
        if c['coattention']:
            log_entry['d2q'] = cPickle.dumps(result['d2q'][0])
            log_entry['q2d'] = cPickle.dumps(result['q2d'][0])
        log[q_id] = log_entry

        if done_examples % 100 == 0:
            print_stats()
    print_stats()

    with open(log_path, 'w') as dst:
        json.dump(log, dst, indent=2, sort_keys=True)
    with open(dest_path, 'w') as dst:
        json.dump(predictions, dst, indent=2, sort_keys=True)
