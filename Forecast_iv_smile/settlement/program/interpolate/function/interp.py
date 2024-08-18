import numpy as np
import matplotlib.pyplot as plt

#自己寫的三階樣條插值函數
def idx_location(x, y):
    y = np.reshape(y, (-1, 1))
    idx = np.arange(len(x))
    idx = np.tile(idx, len(y))
    idx = np.arange(len(x))
    Idx = np.tile(idx, (len(y), 1))
    match_ = np.less_equal(x, y)
    max_less_y_idx = np.max(match_ * Idx, axis=1)
    return max_less_y_idx

class cubic_spline_interp:
    def __init__(self, x, y, second_diff_x0 = 0, second_diff_xn = 0):
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        self.x = np.sort(x)
        self.y = y[np.argsort(x)]
        self.second_diff_x0 = second_diff_x0
        self.second_diff_xn = second_diff_xn
    def interp_function(self):
        diff_x = self.x[1:] - self.x[:-1]
        diff_y = self.y[1:] - self.y[:-1]
        slope = diff_y / diff_x

        slope_1 = slope[1:]
        slope_2 = slope[:-1]

        second_diff_x0 = self.second_diff_x0
        second_diff_xn = self.second_diff_xn
        f = np.hstack((second_diff_x0, 3*(slope_1-slope_2), second_diff_xn))
        f = np.reshape(f, (-1, 1))
        n = len(self.x)
        H = np.zeros((n, n))
        H[0, 0] = 1
        H[n-1, n-1] = 1

        for i in range(1, n-1, 1):
            H[i, i-1:i+2] = np.array([diff_x[i-1], 2*(diff_x[i-1] + diff_x[i]), diff_x[i]])

        c = np.linalg.inv(H) @ f

        d = (c[1:]-c[:-1]) / (3*np.reshape(diff_x, (-1, 1)))
        b = np.reshape(slope, (-1,1)) - (1/3)*(2*c[:-1] + c[1:])*np.reshape(diff_x, (-1, 1))
        a = np.reshape(self.y[:-1], (-1, 1))

        c = np.reshape(c[:-1], (-1, 1))
        return np.hstack((a, b, c, d))
    def __call__(self, x_new):
        func = self.interp_function()
        
        x_new = np.reshape(x_new, (-1))
        power = np.reshape(np.repeat(np.arange(4), (len(x_new))), (4, -1))
        max_less_x_new_idx = idx_location(self.x, x_new)
        idx = np.arange(len(max_less_x_new_idx))
        greater_equal_xmax = idx[np.greater_equal(max_less_x_new_idx, len(self.x)-1)]
        max_less_x_new_idx[greater_equal_xmax] = len(self.x)-2
        
        Func = func[max_less_x_new_idx]
        X_new = (x_new - self.x[max_less_x_new_idx]) ** power
 
        return np.diagonal(Func @ X_new)
    
def Bondfilter(x, lowerbond, upperbond):
    condition = np.all(np.vstack((x >= lowerbond, x <= upperbond)), axis=0)
    return x[condition], condition


#內插函數
def interpolate(x, y, point_num):
    if point_num <= len(x):
        return x, y

    x_max = x.max()
    x_min = x.min()
    f = cubic_spline_interp(x, y)
    x_new = np.linspace(x_min, x_max, point_num)
    y_new = f(x_new)
    return x_new, y_new



def interpolate_IncreasePoints(x, y, points):
    points = Bondfilter(points, x.min(), x.max())[0]
    f = cubic_spline_interp(x, y)
    x_new = np.hstack((x, points))
    x_new = np.unique(x_new)
    y_new = f(x_new)
    return x_new, y_new

def interpolate_range(x, y, points, points_num):
    points = np.array(points)
    x, y = interpolate_IncreasePoints(x, y, points)
    x, condition = Bondfilter(x, points.min(), points.max())
    y = y[condition]
    x_new, y_new = interpolate(x, y, points_num)
    return x_new, y_new