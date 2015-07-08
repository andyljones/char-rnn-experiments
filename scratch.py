# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:09:53 2015

@author: andyjones
"""

import scipy as sp

#from keras.models import Sequential
#from keras.layers.recurrent import GRU
#from keras.layers.embedding import Embedding

def make_batch_generator(text, batch_size=50, seq_length=50, active_range=(0, 1)):
    alphabet = sorted(list(set(text)))
    encoding = {c: i for i, c in enumerate(alphabet)}
    
    num_batches = len(text)/batch_size
    lower_bound = batch_size*int(active_range[0]*num_batches)
    upper_bound = batch_size*int(active_range[1]*num_batches)
    text_to_use = text[lower_bound:upper_bound]

    encoded_text = sp.array([encoding[c] for c in text_to_use], dtype=sp.uint8)
    all_batches = sp.array([encoded_text[i:seq_length+i] for i in range(len(encoded_text) - seq_length)])
    shuffled_batches = sp.random.permutation(all_batches)    
    
    epoch_number = 0
    batch_number = 0
    while True:
        batch = shuffled_batches[batch_number*batch_size:(batch_number+1)*batch_size]
        yield epoch_number, batch_number, batch

        if batch_number > len(shuffled_batches)/batch_size - 2:           
            shuffled_batches = sp.random.permutation(shuffled_batches)
            epoch_number += 1
            batch_number = 0
        else:
            batch_number += 1
        
    

#model = Sequential()
#model.add(Embedding())
#model.add(GRU())