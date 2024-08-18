import numpy as np
import pandas as pd
from function.function_black import vectorized_premium
from function.function_select import select_index_batch

expiry = 'NearbyMonth'
IV_Data = pd.read_csv('./../../Data/Organized/IV/{}.csv'.format(expiry), 
                      encoding='Big5', index_col=False)
IV_matrix = np.array(IV_Data)
idx_IV_matrix = np.arange(len(IV_matrix))

column_call_put_name = '買賣權'
column_call_put_index = IV_Data.columns.get_loc(column_call_put_name)

#買權資料在原本資料的哪幾列
rows_call = idx_IV_matrix[np.equal(IV_matrix[:, column_call_put_index], 'call')]
#買權資料的交易日期、到期日期、履約價
DateK_call = IV_matrix[rows_call, :3]

#賣權資料在原本資料的哪幾列
rows_put = idx_IV_matrix[np.equal(IV_matrix[:, column_call_put_index], 'put')]
#賣權資料的交易日期、到期日期、履約價
DateK_put = IV_matrix[rows_put, :3]

#找出擁有相同交易日期、到期日期、履約價的買權與賣權，找出匹配的資料分別位於買權與賣權資料的第幾列
# idx_call_filter即匹配資料為於買權的索引，idx_put_filter即匹配資料為於賣權的索引
#匹配資料代表著：同一交易日到期日與履約價，其同時有買權與賣權
idx_call_filter, idx_put_filter = select_index_batch(DateK_call, DateK_put, 10000, 10000, axis=1)

#匹配資料位於原資料中、為買權((存在同交易日到期日履約價之賣權))的位於原資料哪幾列
rows_call_filter = rows_call[idx_call_filter]
#匹配資料位於原資料中、為賣權((存在同交易日到期日履約價之買權))的位於原資料哪幾列
rows_put_filter = rows_put[idx_put_filter]
#匹配資料位於原資料中、為賣權(存在同交易日到期日履約價之買權)的哪幾列的值為False，其餘為True
not_put_filter = np.ones(len(IV_matrix), dtype=bool)
not_put_filter[rows_put_filter] = False

column_names_reserve = ['交易日期', '到期日期', '履約價', '期貨成交量', '期貨開盤價','期貨最高價', \
                        '期貨最低價', '期貨收盤價', '期貨結算價', '到期天數', '無風險利率']
column_index_reserve = [IV_Data.columns.get_loc(col) for col in column_names_reserve]

IV_matrix_reserve = IV_matrix[not_put_filter, :]
IV_matrix_reserve = IV_matrix_reserve[:, column_index_reserve]

price_types = ['收盤價', '結算價']

for price_type in price_types:
    column_names  = ['成交量', '期貨{}'.format(price_type), '履約價', '買賣權', price_type, \
                 '無風險利率', '到期天數', '隱含波動率({})'.format(price_type)]
    column_index = [IV_Data.columns.get_loc(col) for col in column_names]
    F0 = IV_matrix[:, column_index[1]]
    K = IV_matrix[:, column_index[2]]
    c_p = IV_matrix[:, column_index[3]]
    price = IV_matrix[:, column_index[4]]
    r = IV_matrix[:, column_index[5]]
    T = IV_matrix[:, column_index[6]] / 365
    IV_m = IV_matrix[:, column_index[7]]
    Ep = vectorized_premium(c_p, F0, K, r, IV_m, T)
    error_m = abs(Ep - price)

    rows_correct = idx_IV_matrix[np.less_equal(error_m, 1e-6)] #計算沒問題的資料在原資料哪幾列
    rows_wrong = idx_IV_matrix[np.logical_not(np.less_equal(error_m, 1e-6))] #錯誤的資料在哪幾列

    volume_adjust = (IV_matrix[:, column_index[0]]).copy() #成交量
    volume_adjust[rows_wrong] = 0 #將錯誤有問題的資料的成交量改成0
    iv_adjust = (IV_matrix[:, column_index[7]]).copy() #隱含波動率
    #iv_wrong = iv_adjust_closing[rows_wrong_closing] 可以看看錯誤的隱含波動率長怎樣(理論上都是nan)
    iv_adjust[rows_wrong] = 0 #將有問題的資料的隱含波動率設為0


    #「調整後」的買權的隱含波動率向量
    iv_call_adjust = iv_adjust[rows_call] 
    #「調整後」的買權的成交量向量
    volume_call_adjust = volume_adjust[rows_call]
    #「調整後」的賣權的隱含波動率向量
    iv_put_adjust = iv_adjust[rows_put]
    #「調整後」的賣權的成交量向量
    volume_put_adjust = volume_adjust[rows_put]
    volume_put_filter_adjust = volume_adjust[rows_put_filter]
    
    #調整後的買權隱含波動率的向量，且存在與其相同交易日到期日履約價的賣權
    iv_call_filter_adjust = iv_adjust[rows_call_filter]
    #調整後的賣權隱含波動率的向量，且存在與其相同交易日到期日履約價的買權
    iv_put_filter_adjust = iv_adjust[rows_put_filter]
    #調整後的買權成交量的向量，且存在與其相同交易日到期日履約價的賣權
    volume_call_filter_adjust = volume_adjust[rows_call_filter]
    #調整後的賣權成交量的向量，且存在與其相同交易日到期日履約價的買權
    volume_put_filter_adjust = volume_adjust[rows_put_filter]
    #iv_call_filter_adjust_closing 與 iv_put_filter_adjust_closing 有相同交易日到期日與履約價，只是買賣權不同 
    #volume_call_filter_adjust_closing 與  volume_put_filter_adjust_closing 相同道理


    #買賣權成交量相加
    volume_sum_filter_adjust = volume_call_filter_adjust + volume_put_filter_adjust

    #依照成交量為基礎，將擁有相同履約價、交易日、到期日的買權與賣權的隱含波動率取個加權平均
    weight_iv_call = volume_call_filter_adjust / volume_sum_filter_adjust
    weight_iv_put = volume_put_filter_adjust / volume_sum_filter_adjust
    weight_avg_iv = weight_iv_call*iv_call_filter_adjust + weight_iv_put*iv_put_filter_adjust


    volume_adjust_final = volume_adjust.copy()
    iv_adjust_final = iv_adjust.copy()


    #將調整後的成交量，將那些「存在相同交易日到期日履約價之賣權」的買權的成交量改成買賣權成交量相加
    volume_adjust_final[rows_call_filter] = volume_sum_filter_adjust  
    #刪除掉「存在相同交易日到期日履約價之買權」的賣權的資料
    volume_adjust_final = volume_adjust_final[not_put_filter] 

    #將調整後的隱含波動率，將那些「存在相同交易日到期日履約價之賣權」的買權的隱含波動率改成加權平均的隱含波動率
    iv_adjust_final[rows_call_filter] = weight_avg_iv
    #刪除掉「存在相同交易日到期日履約價之買權」的賣權的資料
    iv_adjust_final = iv_adjust_final[not_put_filter] 


    #將調整後的買賣權，將那些「存在相同交易日到期日履約價之賣權」的買權的改成'call and put'
    c_or_p_adjust = (IV_matrix[:, column_index[3]]).copy()
    c_or_p_adjust[rows_call_filter] = 'call and put'
    #刪除掉「存在相同交易日到期日履約價之買權」的賣權的資料
    c_or_p_adjust = c_or_p_adjust[not_put_filter] 


    CPVIV_matrix_new = (np.vstack((c_or_p_adjust, volume_adjust_final, iv_adjust_final))).T
    IV_matrix_reserve = np.hstack((IV_matrix_reserve, CPVIV_matrix_new))




column_names_new = np.array(['買賣權(收盤價)', '成交量(收盤價)', '隱含波動率(收盤價)', \
                             '買賣權(結算價)', '成交量(結算價)', '隱含波動率(結算價)'])
column_name_all = np.hstack((np.array(column_names_reserve), column_names_new))
IV_Data_new = pd.DataFrame(data=IV_matrix_reserve, columns=column_name_all)
IV_Data_new.to_csv('./../../Data/Organized/MergeIV/{}.csv'.format(expiry), encoding='Big5',\
                   index = False)


print('Finish !!')