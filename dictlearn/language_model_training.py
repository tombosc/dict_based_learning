import os
import pprint
import logging

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

from dictlearn.util import rename
from dictlearn.data import Data
from dictlearn.extensions import DumpTensorflowSummaries
from dictlearn.language_model import LanguageModel
from dictlearn.retrieval import Dictionary

logger = logging.getLogger()


def train_language_model(config, save_path, fast_start):
    new_training_job = False
    if not os.path.exists(save_path):
        logger.info("Start a new job")
        new_training_job = True
        os.mkdir(save_path)
    else:
        logger.info("Continue an existing job")
    main_loop_path = os.path.join(save_path, 'main_loop.tar')

    c = config
    data = Data(c['data_path'], c['layout'], c['top_k_words'])
    dict_ = None
    if c['dict_path']:
        dict_ = Dictionary(c['dict_path'])

    lm = LanguageModel(c['dim'], data.vocab, dict_,
                       c['standalone_def_rnn'],
                       weights_init=Uniform(width=0.1),
                       biases_init=Constant(0.))
    lm.initialize()

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
    last_correct, = VariableFilter(name='last_correct')(cg)
    last_correct_acc = rename(last_correct.mean(), 'last_correct_acc')
    monitored_vars = [cost, last_correct_acc]
    if c['dict_path']:
        num_definitions, = VariableFilter(name='num_definitions')(cg)
        max_definition_length, = VariableFilter(name='max_definition_length')(cg)
        monitored_vars.extend([num_definitions, max_definition_length])

    parameters = cg.get_parameter_dict()
    logger.info("Trainable parameters" + "\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape)
                     for key in sorted(parameters.keys())],
                    width=120))

    algorithm = GradientDescent(
        cost=cost,
        parameters=parameters.values(),
        step_rule=CompositeRule([
            Adam(learning_rate=c['learning_rate']),
            StepClipping(c['grad_clip_threshold'])]))
    extensions = [
        Load(main_loop_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job),
        Timing(every_n_batches=c['mon_freq_train']),
        TrainingDataMonitoring(
            monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
        DataStreamMonitoring(
            monitored_vars,
            data.get_stream('valid', batch_size=c['batch_size_valid']),
            prefix="valid").set_conditions(
                before_training=not fast_start,
                every_n_batches=c['mon_freq_valid']),
        Checkpoint(main_loop_path,
                   before_training=not fast_start,
                   every_n_batches=c['save_freq_batches']),
        DumpTensorflowSummaries(save_path,
                                every_n_batches=c['mon_freq_train']),
        Printing(every_n_batches=c['mon_freq_train']),
    ]
    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'], max_length=c['max_length'])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)
    main_loop.run()
