import numpy

def vec2str(vector):
    return "".join(map(chr, vector)).strip('\00')

def str2vec(str_, length):
    vector = numpy.array(map(ord, str_))[:length]
    pad_length = max(0, length - len(str_))
    return numpy.pad(vector, (0, pad_length), 'constant')

def rename(var, name):
    var.name = name
    return var
