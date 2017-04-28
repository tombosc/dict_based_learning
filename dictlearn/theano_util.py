import numpy

from theano import tensor


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

