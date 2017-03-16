import theano
from theano import tensor

from blocks.initialization import Uniform, Constant
from blocks.algorithms import Adam, GradientDescent
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

from dictlearn.util import rename
from dictlearn.data import Data
from dictlearn.language_model import LanguageModel

def train_language_model(data_path, layout):
    data = Data(data_path, layout)

    lm = LanguageModel(128, data.vocab,
                       weights_init=Uniform(width=0.1),
                       biases_init=Constant(0.))
    lm.initialize()

    words = tensor.ltensor3('lines')
    words_mask = tensor.matrix('lines_mask')
    if theano.config.compute_test_value != 'off':
        test_value_data = next(
            data.get_stream('train', batch_size=4).get_epoch_iterator())
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
        step_rule=Adam(learning_rate=0.001))
    extensions = [Timing(),
                  TrainingDataMonitoring(
                      [cost, last_correct_acc], prefix="train",
                      every_n_batches=10),
                  Printing(every_n_batches=10)]

    main_loop = MainLoop(
        algorithm,
        data.get_stream('train', batch_size=32),
        extensions=extensions)
    main_loop.run()
