import numpy as np


#日期類型
import calendar
import datetime as dt
from datetime import datetime
def ROCtoAD(date): #ex:'11101' ---> '2022/01'
    return str(int(date[:3])+1911) + '/' +  date[3:]
vectorized_ROCtoAD = np.vectorize(ROCtoAD)

def date_tran(date):#可將'2015/1/30'---->'2015/01/30'
    date = datetime.strptime(date, '%Y/%m/%d')

    return date.strftime('%Y/%m/%d')
def thirdWednesday(date):#輸入：'202311' 輸出：'2023/11/15'(2023/11第三個周三的日期)
    first_weekday, days = calendar.monthrange(year=int(date[:4]), month=int(date[4:]))
    if first_weekday > 2:
        first_weekday =  first_weekday - 7
    thirdWed =  1 + (2-first_weekday) + 14
    return date[:4] + '/' + date[4:] + '/' + str(thirdWed) 
vectorized_thirdWednesday = np.vectorize(thirdWednesday)
vectorized_date_tran = np.vectorize(date_tran)


def dates_between_days(date1, date2):#輸入：'2023/11/15', '2023/12/18' 輸出：3 (兩個日期的差距天數，為int) 
    date1 = datetime.strptime(date1, "%Y/%m/%d").date()
    date2 = datetime.strptime(date2, "%Y/%m/%d").date()  
    Days = (date2 - date1).days  
    return Days 
vectorized_betwween_days = np.vectorize(dates_between_days)