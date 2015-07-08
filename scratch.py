# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:09:53 2015

@author: andyjones
"""

import scipy as sp
import logging

from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense
from keras.layers.recurrent import GRU, LSTM
from keras.optimizers import RMSprop, adam
from keras.initializations import uniform
from keras.layers.convolutional import Convolution1D

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

    def make_generator():
        shuffled_batches = sp.random.permutation(all_batches)    
        
        epoch_count = 0
        batch_count = 0
        while True:
            batch = shuffled_batches[batch_count*batch_size:(batch_count+1)*batch_size]
            one_hot_batch = encode_one_hot(batch, len(alphabet))
            X_batch = one_hot_batch[:, :-1]
            Y_batch = one_hot_batch[:, 1:]
            yield epoch_count, batch_count, X_batch, Y_batch

            if batch_count > len(shuffled_batches)/batch_size - 2:           
                shuffled_batches = sp.random.permutation(shuffled_batches)
                epoch_count += 1
                batch_count = 0
            else:
                batch_count += 1
                
    generator = make_generator()                
           
    return generator, encoder

def make_model(alphabet_size=65, seq_length=50, layer_size=128):
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
    
def make_test_model_gen(model, batch_gen):
    total_loss = 0.
    total_seen = 0
    last_epoch = 0
    while True:
        epoch, batch, X_batch, Y_batch = next(batch_gen)
        
        if epoch != last_epoch:
            print('Epoch {}, testing loss {:4f}'.format(epoch, loss))   
            yield epoch
            
            last_epoch = epoch
            total_loss = 0.
            total_seen = 0
        
        loss = model.test_on_batch(X_batch, Y_batch)[()]
        total_loss += loss
        total_seen += len(X_batch)
        
def make_train_model_gen(model, batch_gen, test_interval):
    batches_seen = 0
    last_epoch = 0
    while True:
        epoch, batch, X_batch, Y_batch = next(batch_gen)
        
        if epoch != last_epoch:
            last_epoch = epoch
            batches_seen = 0
        if batches_seen % test_interval == 0:
            yield epoch
            
        loss = model.train_on_batch(X_batch, Y_batch)[()]
        print('Epoch {}, batch {}, training loss {:4f}'.format(epoch, batch, loss))        
        batches_seen += 1        
        
def train_and_test_model(model, text, total_epochs=100, test_interval=1000):
    train_batch_gen, _ = make_batch_generator(text, active_range=(0., 0.95))  
    train_model_gen = make_train_model_gen(model, train_batch_gen, test_interval=test_interval)

    test_batch_gen, _ = make_batch_generator(text, active_range=(0.95, 1.))
    test_model_gen = make_test_model_gen(model, test_batch_gen)
    
    epoch = 0
    while epoch < total_epochs:
        epoch = next(train_model_gen)
        next(test_model_gen)
        
    
        
        