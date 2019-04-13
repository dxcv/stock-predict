import tushare as ts

df = ts.get_h_data('sz50',start='2006-01-01',end='2017-12-31')

df.to_csv('C:/Users/jacky/Desktop/XMU/HMM/sz501.csv')