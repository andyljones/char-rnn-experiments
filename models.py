# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:50:40 2015

@author: andy
"""
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import RMSprop, adam
from keras.initializations import uniform

from initializations import random_restricted_orthogonal_matrix

import logging
import scipy as sp

def get_number_of_params(model):
    return sum([sp.prod(p.shape.eval()) for p in model.params])

def make_karpathy_lstm(alphabet_size=65, seq_length=50, layer_size=128):
    """This is a work-alike for the default LSTM used in Karpathy's char-rnn work."""
    
    initializer = lambda s: uniform(s, 0.08)

    model = Sequential()
    model.add(LSTM(alphabet_size, 
                   layer_size, 
                   truncate_gradient=seq_length,
                   init=initializer, 
                   inner_init=initializer,
                   inner_activation='sigmoid',
                   activation='tanh',
                   forget_bias_init=initializer,
                   return_sequences=True))
    model.add(LSTM(layer_size, 
                   layer_size, 
                   truncate_gradient=seq_length,
                   init=initializer, 
                   inner_init=initializer,
                   inner_activation='sigmoid',
                   activation='tanh',
                   forget_bias_init=initializer,
                   return_sequences=True))  
    model.add(TimeDistributedDense(layer_size, 
                                   alphabet_size,
                                   init=initializer,
                                   activation='softmax'))
    
    optimizer = RMSprop(lr=2e-3, rho=0.95)
    optimizer.clipnorm = 5
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer)
    
    logging.info('Number of parameters: {}'.format(get_number_of_params(model)))    
        
    return model

def make_rnn(alphabet_size=65, seq_length=50, layer_size=256):
    
    init = lambda s: random_restricted_orthogonal_matrix(s, 0)
    inner_init = lambda s: random_restricted_orthogonal_matrix(s, 128)
    
    model = Sequential()
    model.add(SimpleRNN(alphabet_size,
                        layer_size,
                        truncate_gradient=seq_length,
                        init=init,
                        inner_init=inner_init,
                        activation='relu',
                        return_sequences=True))
    model.add(SimpleRNN(layer_size,
                        layer_size,
                        truncate_gradient=seq_length,
                        init=init,
                        inner_init=inner_init,
                        activation='relu',
                        return_sequences=True))
    model.add(TimeDistributedDense(layer_size,
                                   alphabet_size,
                                   init=init,
                                   activation='softmax'))
                                   
    optimizer = adam()
    optimizer.clipnorm = 5
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer)
                      
    logging.info('Number of parameters: {}'.format(get_number_of_params(model)))   
    
    return model
                