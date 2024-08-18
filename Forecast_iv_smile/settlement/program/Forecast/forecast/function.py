import numpy as np
#斜率、函數移動相關
def Slope(X, Y, axis=1):
    sorted_idx = np.argsort(X, axis=axis)
    X = np.take_along_axis(X, sorted_idx, axis=axis)
    Y = np.take_along_axis(Y, sorted_idx, axis=axis)
    if axis==1:
        delta_X = X[:,1:] - X[:,:-1]
        delta_Y = Y[:,1:] - Y[:,:-1]
    if axis==0:
        delta_X = X[1:] - X[:-1]
        delta_Y = Y[1:] - Y[:-1]
    return delta_Y / delta_X

def recovery_curve(x, slope, y0=0):
    delta_x = x[1:] - x[:-1]
    delta_y = delta_x*slope
    y = np.array([delta_y[:i].sum() for i in range(1, len(x), 1)]) + y0
    y = np.hstack((np.array([y0]), y))
    return y

def min_mse_y0(y, x, slope_yhat):
    n = len(y)
    y_sum = np.sum(y)
    x0 = x[:-1]
    x1 = x[1:]
    x_diff = x1-x0
    k = np.arange(n-1, 0, -1)
    ss = slope_yhat * k * x_diff
    y0 = (y_sum - np.sum(ss)) / n
    return y0

def minSSE_recovery(y, x, slope_yhat):
    y0 = min_mse_y0(y, x, slope_yhat)
    y_hat = recovery_curve(x, slope=slope_yhat, y0=y0)
    SSE = np.sum((y - y_hat) * (y - y_hat))
    return y_hat, SSE

#-------------------------------------------------------------
def TimeSeriesData(X, seq_length, drop_out_columns=[]):
    X_ = np.zeros((len(X)-seq_length, seq_length, len(X[0])))
    y_ = np.zeros((len(X)-seq_length, len(X[0])-len(drop_out_columns)))
    columns = np.arange(len(X[0]))
    reserve_columns = columns[np.logical_not(np.isin(columns, drop_out_columns))]
    for i in range(len(X) - seq_length):
        X_[i] = X[i:i+seq_length]
        y_[i] = (X[i+seq_length, reserve_columns])
    return np.array(X_), np.array(y_)