import numpy

def vec2str(vector):
    return "".join(map(chr, vector)).strip('\00')

def str2vec(str_, length):
    return numpy.pad(numpy.array(map(ord, str_)), (0, length - len(str_)), 'constant')

def rename(var, name):
    var.name = name
    return var
