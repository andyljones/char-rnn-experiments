# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:09:53 2015

@author: andyjones
"""

from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.embedding import Embedding

def make_batch_generator(text, batch_size=50, seq_length=50, active_range=(0, 1)):
    alphabet = sorted(list(set(text)))
    encoding = {c: i for i, c in enumerate(alphabet)}
    
    lower_bound = int(active_range[0]*len(text))
    upper_bound = int(active_range[1]*len(text))
    text_to_use = text[lower_bound:upper_bound]

    encoded_text = [encoding[c] for c in text_to_use]
    
    while True:
        pass

#model = Sequential()
#model.add(Embedding())
#model.add(GRU())