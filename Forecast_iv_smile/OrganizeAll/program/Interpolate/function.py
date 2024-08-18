from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import norm
#Black's model 相關函數
N = norm.cdf
n = norm.pdf
D1 = lambda F0, K, v, T  : (np.log(F0/K)+T*((v**2)/2)) / (v*(T**0.5))
D2 = lambda F0, K, v, T  : (np.log(F0/K)-T*((v**2)/2)) / (v*(T**0.5))
def premium(c_p, F0, K, r, v, T):
    d1 = D1(F0, K, v, T)
    d2 = D2(F0, K, v, T)
    if c_p == 'call':
        c = np.exp(-r*T) * (F0*N(d1) - K*N(d2))
        return c
    if c_p == 'put':
        p = np.exp(-r*T) * (K*N(-d2) - F0*N(-d1))
        return p
vectorized_premium = np.vectorize(premium)

def UperBond(c_p, F0, K, r, T): #價格理論上界
    if c_p == 'call':
        return np.exp(-r*T) * F0
    if c_p == 'put':
        return  np.exp(-r*T) * K
def LowerBond(c_p, F0, K, r, T): #價格理論下界
    if c_p == 'call':
        return max(0, np.exp(-r*T) * (F0 - K))
    if c_p == 'put':
        return max(0, np.exp(-r*T) * (K - F0))
    
#函數向量化
vectorized_premium = np.vectorize(premium)
vectorized_UperBond = np.vectorize(UperBond)
vectorized_LowerBond = np.vectorize(LowerBond)

def Bondfilter(x, lowerbond, upperbond):
    condition = np.all(np.vstack((x >= lowerbond, x <= upperbond)), axis=0)
    return x[condition], condition


#內插函數
def interpolate(x, y, point_num, kind='linear', keep=True):
    if point_num <= len(x):
        return x, y

    
    N = point_num - len(x)
    if keep:
        lines_num = len(x) - 1
        choice_idxs = np.random.choice(range(lines_num), (N%lines_num), replace=False)

        points_num_in_everyintrval = np.array([1 if i in choice_idxs else 0 \
                                       for i in range(lines_num)])
        points_num_in_everyintrval = points_num_in_everyintrval + int(N/lines_num)
        newx = np.zeros(point_num)
        newy = np.zeros(point_num)
        j = 0
        for i in range(lines_num):
            x_ = x[i:i+2]
            y_ = y[i:i+2]
            x_min = x_.min()
            x_max = x_.max()
            f = interp1d(x_, y_, kind=kind)
            new_x = np.linspace(x_min, x_max, points_num_in_everyintrval[i]+2)[:-1]
            new_y = f(new_x)
 
            newx[j:j+points_num_in_everyintrval[i]+1] = new_x
            newy[j:j+points_num_in_everyintrval[i]+1] = new_y
            j = j + points_num_in_everyintrval[i]+1
        newx[-1] = x[-1]
        newy[-1] = y[-1]
        return newx, newy
    else:
        x_max = x.max()
        x_min = x.min()
        f = interp1d(x, y, kind)
        x_new = np.linspace(x_min, x_max, point_num)
        y_new = f(x_new)
        return x_new, y_new



def interpolate_IncreasePoints(x, y, points, kind='linear'):
    points = Bondfilter(points, x.min(), x.max())[0]
    f = interp1d(x, y, kind=kind)
    y_new = f(points)
    x_new = np.hstack((x, points))
    x_new = np.unique(x_new)
    y_new = f(x_new)
    return x_new, y_new

def interpolate_range(x, y, points, points_num, kind='linear', keep=True):
    points = np.array(points)
    x, y = interpolate_IncreasePoints(x, y, points, kind)
    x, condition = Bondfilter(x, points.min(), points.max())
    y = y[condition]
    x_new, y_new = interpolate(x, y, points_num, kind, keep)
    return x_new, y_new
    




