import numpy as np  
import pandas as pd  
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

def Vega(F0, K, r, v, T):
    d1 = D1(F0, K, v, T)
    return F0*(T**0.5)*n(d1)*np.exp(-r*T)
def find_vol(c_p ,price, F0, K, r, T, v0=0.5):
    MAX_ITERATIONS = 100
    PRECISION = 10**(-6)
    blacks = premium
    v = v0

    for i in range(0, MAX_ITERATIONS):
        d = blacks(c_p, F0, K, r, v, T) - price
        vega = Vega(F0, K, r, v, T)
        if (abs(d) < PRECISION):
            break
        v = v - d/vega # f(x) / f'(x)

    return v
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
vectorized_find_vol = np.vectorize(find_vol)
vectorized_premium = np.vectorize(premium)
vectorized_UperBond = np.vectorize(UperBond)
vectorized_LowerBond = np.vectorize(LowerBond)

def IV_table(data, variable_columns_index, V0=[0.5], IV_column='隱含波動率', columns='original',  T_unit='days'):
    if columns == 'original':
        columns = np.hstack((data.columns.to_numpy(), np.array([IV_column])))
    
    IV_matrix_all = np.array(data)
    IV_all = np.array([float('nan')]*len(data))

    
    F0 = np.array(IV_matrix_all[:, variable_columns_index[0]])
    K = np.array(IV_matrix_all[:, variable_columns_index[1]])
    c_p = np.array(IV_matrix_all[:, variable_columns_index[2]])
    price = np.array(IV_matrix_all[:, variable_columns_index[3]])
    r = np.array(IV_matrix_all[:, variable_columns_index[4]])
    if T_unit == 'days':
        y = 365
    elif T_unit == 'month':
        y = 30
    elif T_unit == 'year':
        y = 1
    
    T = np.array(IV_matrix_all[:, variable_columns_index[5]]) / y
    #理論價格上下界
    priceUper = vectorized_UperBond(c_p, F0, K, r, T)
    priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
    correct = (price < priceUper) & (price > priceLower)
    wrong = np.bitwise_not(correct)
    rows = np.where(correct)[0] #定價沒問題的資料位於原資料的哪些行
    #rows為[28, 56, 145]，代表IV_matrix_all中第28、56、145的行是沒問題的
    IV_matrix = IV_matrix_all[correct] #定價沒問題的資料
    wrong_rows = np.where(wrong)[0] #定價有問題的資料位於原資料的哪些行
    #wrong_matrix = IV_matrix_all[wrong] #定價有問題的資料

    for i in range(len(V0)):
        v0 = V0[i]
        F0 = np.array(IV_matrix[:, variable_columns_index[0]])
        K = np.array(IV_matrix[:, variable_columns_index[1]])
        c_p = np.array(IV_matrix[:, variable_columns_index[2]])
        price = np.array(IV_matrix[:, variable_columns_index[3]])
        r = np.array(IV_matrix[:, variable_columns_index[4]])
        T = np.array(IV_matrix[:, variable_columns_index[5]]) / y

        IV = vectorized_find_vol(c_p ,price, F0, K, r, T, v0)
        #驗算：將算好的IV丟回去算出價格與原本的價格比較
        Ep = vectorized_premium(c_p, F0, K, r, IV, T)
        error = abs(Ep - price)
        match_cond = error <= 1e-6
        #得出符合(誤差小於10^-6)的資料在原本表(IV_matrix_all)中的第幾行
        matching_rows = rows[match_cond]
        IV = IV[match_cond]
        IV_all[matching_rows] = IV
       
        #更新：找出驗算不符合的資料，下一次迴圈就用這些資料代入不同v0計算IV
        rows = rows[np.logical_not(match_cond)]
        IV_matrix = IV_matrix[np.logical_not(match_cond)]
        
        #如果再也找不到有問題的資料，就跳出迴圈
        #np.all([True, True]) -----> True
        #np.all([True, True, False, True]) --------> False
        if np.all(match_cond):
            break

    #Error_matrix = IV_matrix
    Error_rows = rows

    IV_matrix_all = np.hstack((IV_matrix_all, np.reshape(IV_all, (-1, 1))))    
    IV_data = pd.DataFrame(data=IV_matrix_all, columns=columns) 

    #輸出為：包含隱含波動率的資料表格，價格不在理論範圍內的資料的索引(行)，
    #    價格在理論範圍內，但算不出其正確隱含波動率的資料的索引(行)，如果不行，可在多找不同的v0再跑程式
    return IV_data, wrong_rows, Error_rows