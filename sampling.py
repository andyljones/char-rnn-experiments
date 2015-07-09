# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:49:43 2015

@author: andy
"""

import scipy as sp

def encode_string(string, encoder):
    return sp.array(map(encoder.__getitem__, string))

def decode_to_string(batch, encoder):
    decoder = {sp.argmax(one_hot): c for c, one_hot in encoder.items()}    
    ints = sp.argmax(batch, 2)
    
    chars = sp.vectorize(decoder.__getitem__)(ints)
    strings = [''.join(row) for row in chars]
    
    return strings
    
def mle_sample(model, encoder, sample_length=100, seq_length=50):
    most_recent_text = ' '*seq_length
    generated_text = ''
    
    for _ in range(sample_length):
        X = encode_string(most_recent_text, encoder)
        Y = model.predict(X[None, :, :])
        S = decode_to_string(Y, encoder)[0]
        s = S[-1]
        
        most_recent_text = most_recent_text[1:] + s
        generated_text = generated_text + s        
        
    return generated_text