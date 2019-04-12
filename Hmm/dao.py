# -*- coding: utf-8 -*-
# Data Access
# 数据接口层
import tushare as ts
import pandas as pd
import numpy as np
import datetime


# 获取指定日期范围的本地数据
# s_year, e_year: 'yyyymmdd' or 'yyyy-mm-dd' str
def get_local_bar_data(code, s_date, e_date):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    df = pd.read_csv('data/' + code + '.csv', index_col='date', date_parser=dateparse)
    df.sort_index(inplace=True)
    return df[s_date:e_date]


# 获取指数的bar数据
def get_bar_data(code, s_date, e_date, is_index):
    return ts.get_h_data(code, start=s_date, end=e_date, index=is_index)


# 数据本地化
# s_year, e_year: yyyy int
def localize_hist_data(code, s_year, e_year):
    df = ts.get_h_data(code, index=True, start=str(s_year) + '-01-01', end=str(s_year) + '-12-31')
    for year in range(s_year + 1, e_year, 1):
        tmp_df = ts.get_h_data(code, index=True, start=str(year) + '-01-01', end=str(year) + '-12-31')
        df = df.append(tmp_df)
    tmp_df = ts.get_h_data(code, index=True, start=str(e_year) + '-01-01',
                           end=datetime.datetime.now().strftime('%Y-%m-%d'))
    df = df.append(tmp_df)
    df.sort_index(inplace=True)
    # 计算日内收益率((当日收盘价-当日开盘价)/当日开盘价)
    df['intraday_return'] = \
        (np.array(df['close']) - np.array(df['open'])) / np.array(df['open'])
    # 计算每日收益率((当日收盘价-昨日收盘价)/昨日收盘价)
    daily_return = \
        (np.array(df['close'])[1:] - np.array(df['close'])[:-1]) / np.array(df['close'][:-1])
    daily_return = np.append(0.0, daily_return)
    df['daily_return'] = daily_return
    open_return = (np.array(df['open'])[1:] - np.array(df['close'])[:-1]) / np.array(df['close'][:-1])
    open_return = np.append(0.0, open_return)
    df['open_return'] = open_return
    df.to_csv(path_or_buf='data/' + code + '.csv', mode='w', header=True)


def localize_shibor_data():
    shibor_df = ts.shibor_data(2006)
    for y in range(2007, 2018):
        df = ts.shibor_data(y)
        shibor_df = shibor_df.append(df)
    shibor_daily_change_rate = \
        (np.array(shibor_df['ON'])[1:] - np.array(shibor_df['ON'])[:-1]) / np.array(shibor_df['ON'])[:-1]
    shibor_daily_change_rate = np.append(0.0, shibor_daily_change_rate)
    shibor_df['daily_change_rate'] = shibor_daily_change_rate
    shibor_df.to_csv(path_or_buf='data/shibor.csv', header=True, index=False)


# 数据本地化
def main():
    code = '000300'
    # code = '000016'
    year = int(datetime.datetime.now().strftime('%Y'))
    # localize_shibor_data()
    localize_hist_data(code, s_year=2010, e_year=year)
    # get_local_bar_data(code, '2010-01-01', '2011-12-31')


if __name__ == "__main__":
    main()
