import numpy

from theano import tensor


import theano.tensor as T
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.scan_module.scan_op import Scan
from toolz import unique
from blocks.config import config


def apply_dropout(var, drop_prob, rng=None,
        seed=None, custom_divisor=None):
    if not rng and not seed:
        seed = config.default_seed
    if not rng:
        rng = MRG_RandomStreams(seed)
    if custom_divisor is None:
        divisor = (1 - drop_prob)
    else:
        divisor = custom_divisor

    return var * rng.binomial(var.shape, p=1 - drop_prob, dtype=theano.config.floatX) / divisor


def parameter_stats(parameters, algorithm):
    vars_ = []
    for name, param in parameters.items():
        num_elements = numpy.product(param.get_value().shape)
        norm = param.norm(2) / num_elements ** 0.5
        grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
        step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
        stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
        stats.name = name + '_stats'
        vars_.append(stats)
    return vars_


def unk_ratio(words, mask, unk):
    num_unk = (tensor.eq(words, unk) * mask).sum()
    return num_unk / mask.sum()
