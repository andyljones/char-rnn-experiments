# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:50:40 2015

@author: andy
"""
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.initializations import uniform

def make_karpathy_lstm(alphabet_size=65, seq_length=50, layer_size=128):
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
    
    return model
    