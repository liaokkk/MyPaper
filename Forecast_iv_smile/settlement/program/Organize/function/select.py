import numpy as np
import math

def select_index(A, B, axis=1):
    X = A.copy()
    Y = B.copy()
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
        Y = np.reshape(Y, (-1, 1))
        axis = 1
    if axis==0:
        X = X.T
        Y = Y.T

    ly = len(Y)
    lx = len(X)
    X_ = np.repeat(X, repeats=ly, axis=0)
    Y_ = np.tile(Y, (lx, 1))
    match = np.equal(X_, Y_)
    match = np.all(match, axis=1)


    idx_Y = np.arange(ly)
    idx_Y = np.tile(idx_Y, (lx))

    idx_X = np.arange(lx)
    idx_X = np.repeat(idx_X, repeats=ly)
    return idx_X[match], idx_Y[match]

def select_index_batch(A, B, X_batchsize, Y_batchsize, axis=1):
    X = A.copy()
    Y = B.copy()
    
    
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
        Y = np.reshape(Y, (-1, 1))
        axis=1
    if axis == 0 :
        X = X.T
        Y = Y.T
    Iteration_X = math.ceil(len(X)/ X_batchsize)
    Iteration_Y = math.ceil(len(Y)/ Y_batchsize)
    idx_X = np.zeros((max(len(X), len(Y)))) - 1
    idx_Y = np.zeros((max(len(X), len(Y)))) - 1 
    for i in range(Iteration_X):
        X_batch = X[i*X_batchsize:(i+1)*X_batchsize]
        for j in range(Iteration_Y):
            Y_batch = Y[j*Y_batchsize:(j+1)*Y_batchsize]
            idx_X_batch, idx_Y_batch = select_index(X_batch, Y_batch, axis=1)
            idx_X_batch = idx_X_batch + i*X_batchsize
            idx_Y_batch = idx_Y_batch + j*Y_batchsize
            idx_X[j*Y_batchsize: j*Y_batchsize + len(idx_X_batch)] = idx_X_batch
            idx_Y[j*Y_batchsize: j*Y_batchsize + len(idx_Y_batch)] = idx_Y_batch
    idx_X = idx_X[np.not_equal(idx_X, -1)]
    idx_Y = idx_Y[np.not_equal(idx_Y, -1)]
    idx_X = idx_X.astype(int)
    idx_Y = idx_Y.astype(int)

    return idx_X, idx_Y



