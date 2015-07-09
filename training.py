# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:52:01 2015

@author: andy
"""
import logging
import interactive_console_options

from models import *
from batch_generation import make_batch_generator
        
def _make_train_model_coroutine(model, batch_gen):
    while True:
        epoch, batch, X_batch, Y_batch = next(batch_gen)
        
        if batch == 0:
            yield
            
        loss = model.train_on_batch(X_batch, Y_batch)[()]
        
        if batch % 10 == 0: 
            logging.info('Epoch {}, batch {}, training loss {:4f}'.format(epoch, batch, loss))          

def _make_test_model_coroutine(model, batch_gen):
    total_loss = 0.
    total_seen = 0
    while True:
        _, _, X_batch, Y_batch = next(batch_gen)
        
        if batch == 0:
            logging.info('Testing loss {:4f}'.format(total_loss/total_seen))   

            total_loss = 0.
            total_seen = 0                 
            
            yield
               
        loss = model.test_on_batch(X_batch, Y_batch)[()]
        total_loss += loss
        total_seen += 1          
        
def train_and_test_model(model, text, total_epochs=20):
    train_batch_gen, _ = make_batch_generator(text, active_range=(0., 0.95))  
    train_model_coroutine = _make_train_model_coroutine(model, train_batch_gen)

    test_batch_gen, _ = make_batch_generator(text, active_range=(0.95, 1.))
    test_model_coroutine = _make_test_model_coroutine(model, test_batch_gen)
    
    for _ in range(total_epochs):
        next(train_model_coroutine)
        next(test_model_coroutine)
        