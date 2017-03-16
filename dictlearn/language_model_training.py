import theano
from theano import tensor

from blocks.initialization import Uniform, Constant
from blocks.algorithms import Adam, GradientDescent
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

from dictlearn.util import rename
from dictlearn.data import Data
from dictlearn.language_model import LanguageModel

def train_language_model(data_path, layout, fast_start, config, save_path):
    data = Data(data_path, layout)
    c = config

    lm = LanguageModel(c['dim'], data.vocab,
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

    cg = ComputationGraph(cost)
    last_correct, = VariableFilter(name='last_correct')(cg)
    last_correct_acc = rename(last_correct.mean(), 'last_correct_acc')


    algorithm = GradientDescent(
        cost=cost,
        parameters=cg.parameters,
        step_rule=Adam(learning_rate=c['learning_rate']))
    extensions = [
        Timing(every_n_batches=c['mon_freq_train']),
        TrainingDataMonitoring(
            [cost, last_correct_acc], prefix="train",
            every_n_batches=c['mon_freq_train']),
        DataStreamMonitoring(
            [cost, last_correct_acc],
            data.get_stream('valid', batch_size=c['batch_size_valid']),
            prefix="valid").set_conditions(
                before_training=not fast_start,
                every_n_batches=c['mon_freq_valid']),
        Printing(every_n_batches=c['mon_freq_train']),
        Checkpoint(save_path,
                   before_training=not fast_start,
                   every_n_batches=c['save_freq_batches'])
    ]
    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'], max_length=c['max_length'])

    main_loop = MainLoop(
        algorithm,
        training_stream,
        extensions=extensions)
    main_loop.run()
