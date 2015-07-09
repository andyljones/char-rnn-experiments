# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:49:43 2015

@author: andy
"""

import scipy as sp

def decode_one_hot(one_hot):
    return sp.argmax(one_hot, 2)

def decode_to_string(batch, encoding):
    decoder = {i: c for c, i in encoding.iteritems()}
    return [''.join(row) for row in sp.vectorize(decoder.__getitem__)(batch)]