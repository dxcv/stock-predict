import tushare as ts

df = ts.get_k_data('sz50',start='2010-01-01',end='2017-12-31')
df.to_csv('./test/sz50.csv')
