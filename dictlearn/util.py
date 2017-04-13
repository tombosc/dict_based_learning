from __future__ import division

import socket

import numpy

def softmax(v, T):
    exp_v = numpy.exp(v/T)
    return exp_v / numpy.sum(exp_v)

def vec2str(vector):
    """Transforms a fixed size vector into a unicode string."""
    return u"".join(map(unichr, vector)).strip('\00')


def str2vec(str_, length):
    """Trasforms a string into a fixed size numpy.array

    Adds padding, if necessary. Truncates, if necessary.

    Importanty, if the input is a unicode string, the resulting
    array with contain unicode codes.

    """
    vector = numpy.array(map(ord, str_))[:length]
    pad_length = max(0, length - len(str_))
    return numpy.pad(vector, (0, pad_length), 'constant')


def rename(var, name):
    var.name = name
    return var


def smart_sum(x):
    for i in range(x.ndim):
        x = x.sum(axis=-1)
    return x


def masked_root_mean_square(x, mask):
    """Masked root mean square for a 3D tensor"""
    return (smart_sum((x * mask[:, :, None]) ** 2) / x.shape[2] / mask.sum()) ** 0.5


def get_free_port():
    # Copy-paste from
    # http://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port
