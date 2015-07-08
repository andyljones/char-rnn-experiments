# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:09:53 2015

@author: andyjones
"""

import scipy as sp

from keras.models import Sequential
from keras.layers.core import Dense, RepeatVector
from keras.layers.recurrent import GRU

def encode_one_hot(batch, alphabet_size):
    batch_is, batch_js = sp.indices((batch.shape[0], batch.shape[1]))  
    one_hot_batch = sp.zeros((batch.shape[0], batch.shape[1], alphabet_size))
    one_hot_batch[batch_is, batch_js, batch] = 1.

    return one_hot_batch

def decode_one_hot(one_hot):
    return sp.argmax(one_hot, 2)

def decode_to_string(batch, encoding):
    decoder = {i: c for c, i in encoding.iteritems()}
    return [''.join(row) for row in sp.vectorize(decoder.__getitem__)(batch)]

def make_batch_generator(text, batch_size=50, seq_length=50, active_range=(0, 1)):
    alphabet = sorted(list(set(text)))
    encoder = {c: i for i, c in enumerate(alphabet)}
    
    num_batches = len(text)/batch_size
    lower_bound = batch_size*int(active_range[0]*num_batches)
    upper_bound = batch_size*int(active_range[1]*num_batches)
    text_to_use = text[lower_bound:upper_bound]

    encoded_text = sp.array([encoder[c] for c in text_to_use], dtype=sp.uint8)
    all_batches = sp.array([encoded_text[i:seq_length+1+i] for i in range(len(encoded_text) - seq_length - 1)])

    def make_internal_generator():
        shuffled_batches = sp.random.permutation(all_batches)    
        
        epoch_number = 0
        batch_number = 0
        while True:
            batch = shuffled_batches[batch_number*batch_size:(batch_number+1)*batch_size]
            one_hot_batch = encode_one_hot(batch, len(alphabet))
            X_batch = one_hot_batch[:, :-1]
            Y_batch = one_hot_batch[:, [-1]]
            yield epoch_number, batch_number, X_batch, Y_batch

            if batch_number > len(shuffled_batches)/batch_size - 2:           
                shuffled_batches = sp.random.permutation(shuffled_batches)
                epoch_number += 1
                batch_number = 0
            else:
                batch_number += 1
                
    return make_internal_generator(), encoder

def make_model(alphabet_size=34, seq_length=50, layer_size=128):
    model = Sequential()
    model.add(GRU(layer_size, layer_size, truncate_gradient=seq_length))
    model.add(Dense(layer_size, alphabet_size, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model