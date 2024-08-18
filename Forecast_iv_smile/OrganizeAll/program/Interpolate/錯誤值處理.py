import pandas as pd
import numpy as np
from function import interpolate_IncreasePoints, vectorized_LowerBond, vectorized_UperBond
top_path = './../../'
IV_path = top_path + 'Data/Organized/IV/'
expiry = 'NearbyMonth'
expiry_path = IV_path + expiry + '/'
IV_data = pd.read_csv(expiry_path + 'Wrong.csv', index_col=False, encoding='Big5')

column_names = ['交易日期', '履約價', '隱含波動率(收盤價)', '買賣權']
column_index = [IV_data.columns.get_loc(col) for col in column_names]


IV_matrix  = np.array(IV_data)


rows = np.arange(len(IV_matrix))

opt_types = ['call', 'put']
variable_columns_names  = ['期貨收盤價', '履約價', '買賣權', '收盤價', '無風險利率', '到期天數']
variable_columns_index = [IV_data.columns.get_loc(col) for col in variable_columns_names]
for opt_type in opt_types:
    rows_type = rows[np.equal(IV_matrix[:, column_index[3]], opt_type)]
    IV_matrix_type = IV_matrix[rows_type]

    iv_col_type = IV_matrix_type[:, column_index[2]].astype(float)
    F0 = np.array(IV_matrix_type[:, variable_columns_index[0]])
    K = np.array(IV_matrix_type[:, variable_columns_index[1]])
    c_p = np.array(IV_matrix_type[:, variable_columns_index[2]])
    price = np.array(IV_matrix_type[:, variable_columns_index[3]])
    r = np.array(IV_matrix_type[:, variable_columns_index[4]])

    T = np.array(IV_matrix_type[:, variable_columns_index[5]]) / 365
    #理論價格上下界
    priceUper = vectorized_UperBond(c_p, F0, K, r, T)
    priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
    correct = (price < priceUper) & (price > priceLower)
    wrong = np.bitwise_not(correct)
    rows_type_wrong = rows_type[wrong]
 
    dates = IV_matrix_type[:, column_index[0]]
    dates = dates[wrong]
    dates = np.unique(dates)

    new_iv_idx = 0
    for date in dates:
        rows_type_day = rows_type[np.equal(IV_matrix_type[:, column_index[0]], date)]
        oneday_type = IV_matrix[rows_type_day]

        iv_day_type = oneday_type[:, column_index[2]].astype(float)
        F0 = np.array(oneday_type[:, variable_columns_index[0]])
        K = np.array(oneday_type[:, variable_columns_index[1]])
        c_p = np.array(oneday_type[:, variable_columns_index[2]])
        price = np.array(oneday_type[:, variable_columns_index[3]])
        r = np.array(oneday_type[:, variable_columns_index[4]])

        T = np.array(oneday_type[:, variable_columns_index[5]]) / 365
        #理論價格上下界
        priceUper = vectorized_UperBond(c_p, F0, K, r, T)
        priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
        correct = (price < priceUper) & (price > priceLower)
        wrong = np.bitwise_not(correct)
        i = 0
        while wrong[i]:
            i = i+1
        j = len(wrong)-1
        while wrong[j]:
            j = j-1
        rows_type_day = rows_type_day[i:j+1]
     

        oneday_correct_type = oneday_type[correct]
        oneday_wrong_type = oneday_type[wrong]
        K_correct_day_type = oneday_correct_type[:, column_index[1]].astype(int)
        K_wrong_day_type = oneday_wrong_type[:, column_index[1]].astype(int)
        iv_correct_day_type = oneday_correct_type[:, column_index[2]].astype(float)
        K_day_type, new_iv_day_type = interpolate_IncreasePoints(x=K_correct_day_type, 
                                                         y=iv_correct_day_type, 
                                                         points=K_wrong_day_type)
        IV_matrix[rows_type_day, column_index[2]] = new_iv_day_type
F0 = np.array(IV_matrix[:, variable_columns_index[0]])
K = np.array(IV_matrix[:, variable_columns_index[1]])
c_p = np.array(IV_matrix[:, variable_columns_index[2]])
price = np.array(IV_matrix[:, variable_columns_index[3]])
r = np.array(IV_matrix[:, variable_columns_index[4]])

T = np.array(IV_matrix[:, variable_columns_index[5]]) / 365
#理論價格上下界
priceUper = vectorized_UperBond(c_p, F0, K, r, T)
priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
correct = (price < priceUper) & (price > priceLower)
wrong = np.bitwise_not(correct)


IV_matrix = IV_matrix[correct]


#--------------------------------------------------------------------------------]

rows = np.arange(len(IV_matrix))
column_names = ['交易日期', '履約價', '隱含波動率(結算價)', '買賣權']
column_index = [IV_data.columns.get_loc(col) for col in column_names]
opt_types = ['call', 'put']
variable_columns_names  = ['期貨結算價', '履約價', '買賣權', '結算價', '無風險利率', '到期天數']
variable_columns_index = [IV_data.columns.get_loc(col) for col in variable_columns_names]
for opt_type in opt_types:
    rows_type = rows[np.equal(IV_matrix[:, column_index[3]], opt_type)]
    IV_matrix_type = IV_matrix[rows_type]

    iv_col_type = IV_matrix_type[:, column_index[2]].astype(float)
    F0 = np.array(IV_matrix_type[:, variable_columns_index[0]])
    K = np.array(IV_matrix_type[:, variable_columns_index[1]])
    c_p = np.array(IV_matrix_type[:, variable_columns_index[2]])
    price = np.array(IV_matrix_type[:, variable_columns_index[3]])
    r = np.array(IV_matrix_type[:, variable_columns_index[4]])

    T = np.array(IV_matrix_type[:, variable_columns_index[5]]) / 365
    #理論價格上下界
    priceUper = vectorized_UperBond(c_p, F0, K, r, T)
    priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
    correct = (price < priceUper) & (price > priceLower)
    wrong = np.bitwise_not(correct)
    rows_type_wrong = rows_type[wrong]
 
    dates = IV_matrix_type[:, column_index[0]]
    dates = dates[wrong]
    dates = np.unique(dates)

    new_iv_idx = 0
    for date in dates:
        rows_type_day = rows_type[np.equal(IV_matrix_type[:, column_index[0]], date)]
        oneday_type = IV_matrix[rows_type_day]

        iv_day_type = oneday_type[:, column_index[2]].astype(float)
        F0 = np.array(oneday_type[:, variable_columns_index[0]])
        K = np.array(oneday_type[:, variable_columns_index[1]])
        c_p = np.array(oneday_type[:, variable_columns_index[2]])
        price = np.array(oneday_type[:, variable_columns_index[3]])
        r = np.array(oneday_type[:, variable_columns_index[4]])

        T = np.array(oneday_type[:, variable_columns_index[5]]) / 365
        #理論價格上下界
        priceUper = vectorized_UperBond(c_p, F0, K, r, T)
        priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
        correct = (price < priceUper) & (price > priceLower)
        wrong = np.bitwise_not(correct)
        i = 0
        while wrong[i]:
            i = i+1
        j = len(wrong)-1
        while wrong[j]:
            j = j-1
        rows_type_day = rows_type_day[i:j+1]
     

        oneday_correct_type = oneday_type[correct]
        oneday_wrong_type = oneday_type[wrong]
        K_correct_day_type = oneday_correct_type[:, column_index[1]].astype(int)
        K_wrong_day_type = oneday_wrong_type[:, column_index[1]].astype(int)
        iv_correct_day_type = oneday_correct_type[:, column_index[2]].astype(float)
        K_day_type, new_iv_day_type = interpolate_IncreasePoints(x=K_correct_day_type, 
                                                         y=iv_correct_day_type, 
                                                         points=K_wrong_day_type)
        IV_matrix[rows_type_day, column_index[2]] = new_iv_day_type
F0 = np.array(IV_matrix[:, variable_columns_index[0]])
K = np.array(IV_matrix[:, variable_columns_index[1]])
c_p = np.array(IV_matrix[:, variable_columns_index[2]])
price = np.array(IV_matrix[:, variable_columns_index[3]])
r = np.array(IV_matrix[:, variable_columns_index[4]])

T = np.array(IV_matrix[:, variable_columns_index[5]]) / 365
#理論價格上下界
priceUper = vectorized_UperBond(c_p, F0, K, r, T)
priceLower = vectorized_LowerBond(c_p, F0, K, r, T)
correct = (price < priceUper) & (price > priceLower)
wrong = np.bitwise_not(correct)


IV_matrix = IV_matrix[correct]

IV_data_new = pd.DataFrame(data=IV_matrix, columns=IV_data.columns.to_numpy())
IV_data_new.to_csv(expiry_path + 'All.csv', index=False, encoding='Big5')
print('Finish')