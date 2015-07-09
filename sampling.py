# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:49:43 2015

@author: andy
"""

import scipy as sp

def decode_to_string(batch, encoder):
    decoder = {sp.argmax(one_hot): c for c, one_hot in encoder.items()}    
    ints = sp.argmax(batch, 2)
    
    chars = sp.vectorize(decoder.__getitem__)(ints)
    strings = [''.join(row) for row in chars]
    
    return strings