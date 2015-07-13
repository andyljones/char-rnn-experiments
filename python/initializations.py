# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:35:13 2015

@author: andy
"""
from keras.utils.theano_utils import sharedX

import scipy as sp

def random_orthogonal_matrix(n):
    gaussians = sp.random.normal(size=n*n).reshape((n, n))
    q, _ = sp.linalg.qr(gaussians)
    
    return q
    
def random_restricted_orthogonal_matrix(shape, c):
    m, n = (shape[0], sp.prod(shape[1:]))
    
    U = random_orthogonal_matrix(m - c)

    D = sp.zeros((m - c, n - c))
    D[sp.diag_indices(min(m,n) - c)] = 1.
    
    V = random_orthogonal_matrix(n - c)

    M = sp.eye(m, n)
    M[c:, c:] = U.dot(D).dot(V)
    
    return sharedX(M)