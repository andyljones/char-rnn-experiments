# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:49:07 2015

@author: andy
"""

import scipy as sp

def encode_one_hot(batch, alphabet_size):
    batch_is, batch_js = sp.indices((batch.shape[0], batch.shape[1]))  
    one_hot_batch = sp.zeros((batch.shape[0], batch.shape[1], alphabet_size))
    one_hot_batch[batch_is, batch_js, batch] = 1.

    return one_hot_batch
    
def _make_random_seqs(encoded_text, seq_length):
    actual_seq_length = seq_length + 1
    lower = sp.random.randint(actual_seq_length)
    num_seqs = (len(encoded_text) - lower)/actual_seq_length 
    upper = actual_seq_length*num_seqs + lower
    
    text_to_use = encoded_text[lower:upper]    
    all_seqs = text_to_use.reshape((-1, seq_length+1))

    return sp.random.permutation(all_seqs)

def _make_batch_generator(encoded_text, batch_size, seq_length, alphabet_size):
    shuffled_seqs = _make_random_seqs(encoded_text, seq_length)    
    
    epoch_count = 0
    batch_count = 0
    while True:
        batch = shuffled_seqs[batch_count*batch_size:(batch_count+1)*batch_size]
        one_hot_batch = encode_one_hot(batch, alphabet_size)
        X_batch = one_hot_batch[:, :-1]
        Y_batch = one_hot_batch[:, 1:]
        yield epoch_count, batch_count, X_batch, Y_batch

        if batch_count > len(shuffled_seqs)/batch_size - 2:           
            shuffled_seqs = _make_random_seqs(encoded_text, seq_length)
            epoch_count += 1
            batch_count = 0
        else:
            batch_count += 1

def make_batch_generator(text, batch_size=50, seq_length=50, active_range=(0, 1)):
    alphabet = sorted(list(set(text)))
    encoder = {c: i for i, c in enumerate(alphabet)}
    
    lower_bound = int(active_range[0]*len(text))
    upper_bound = int(active_range[1]*len(text))
    text_to_use = text[lower_bound:upper_bound]
    encoded_text = sp.array([encoder[c] for c in text_to_use], dtype=sp.uint8)
    
    generator = _make_batch_generator(encoded_text, batch_size, seq_length, len(alphabet))                
           
    return generator, encoder