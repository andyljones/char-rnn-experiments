# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:49:07 2015

@author: andy
"""

import scipy as sp

def encode_text(text):
    alphabet = set(list(text))
    char_to_int = {c: i for i, c in enumerate(alphabet)}

    char_to_one_hot = {}
    for c, i in char_to_int.items():
        one_hot = sp.zeros(len(alphabet))
        one_hot[i] = 1.
        char_to_one_hot[c] = one_hot
        
    encoded_text = sp.array(map(char_to_one_hot.__getitem__, text))

    return encoded_text, char_to_one_hot
    
def _make_random_seqs(encoded_text, seq_length):
    actual_seq_length = seq_length + 1
    lower = sp.random.randint(actual_seq_length)
    num_seqs = (len(encoded_text) - lower)/actual_seq_length 
    upper = actual_seq_length*num_seqs + lower
    
    text_to_use = encoded_text[lower:upper]    
    all_seqs = text_to_use.reshape((-1, seq_length+1, encoded_text.shape[-1]))

    return sp.random.permutation(all_seqs)

def _make_batch_generator(encoded_text, batch_size, seq_length, alphabet_size):
    shuffled_seqs = _make_random_seqs(encoded_text, seq_length)    
    
    epoch_count = 0
    batch_count = 0
    while True:
        batch = shuffled_seqs[batch_count*batch_size:(batch_count+1)*batch_size]
        X_batch = batch[:, :-1]
        Y_batch = batch[:, 1:]
        yield epoch_count, batch_count, X_batch, Y_batch

        if batch_count > len(shuffled_seqs)/batch_size - 2:           
            shuffled_seqs = _make_random_seqs(encoded_text, seq_length)
            epoch_count += 1
            batch_count = 0
        else:
            batch_count += 1

def make_batch_generator(text, batch_size=50, seq_length=50, active_range=(0, 1)):
    encoded_text, encoder = encode_text(text)
    
    lower_bound = int(active_range[0]*len(text))
    upper_bound = int(active_range[1]*len(text))
    text_to_use = encoded_text[lower_bound:upper_bound]
    
    generator = _make_batch_generator(text_to_use, batch_size, seq_length, len(encoder))                
           
    return generator, encoder