# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:52:01 2015

@author: andy
"""

from models import *
from batch_generation import make_batch_generator

def _make_test_model_coroutine(model, batch_gen):
    total_loss = 0.
    total_seen = 0
    last_epoch = 0
    while True:
        epoch, batch, X_batch, Y_batch = next(batch_gen)
        
        loss = model.test_on_batch(X_batch, Y_batch)[()]
        total_loss += loss
        total_seen += len(X_batch)        
        
        if epoch != last_epoch:
            print('Testing loss {:4f}'.format(epoch, loss))   
            yield epoch
            
            last_epoch = epoch
            total_loss = 0.
            total_seen = 0
        
def _make_train_model_coroutine(model, batch_gen, test_interval):
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
    train_model_gen = _make_train_model_coroutine(model, train_batch_gen, test_interval=test_interval)

    test_batch_gen, _ = make_batch_generator(text, active_range=(0.95, 1.))
    test_model_gen = _make_test_model_coroutine(model, test_batch_gen)
    
    epoch = 0
    while epoch < total_epochs:
        epoch = next(train_model_gen)
        next(test_model_gen)
        