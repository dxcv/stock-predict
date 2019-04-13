import quandl  # stock data
import pandas as pd  # dataframe
import numpy as np  # array and math functions
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

# set print format-------------------------------------------------------------------
pd.set_option('display.max_rows', 500)  # 最大行数
pd.set_option('display.max_columns', 500)  # 最大列数
pd.set_option('display.width', 4000)  # 页面宽度

# get data-------------------------------------------------------------------
df = quandl.get("WIKI/AMZN")  # get stock data
print('\nstock-data:\n', df.tail())
df = df[['Adj. Close']]  # only take this column

forecast_out = int(30)  # predicting for 30 days into the future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)  # Label column

X = np.array(df.drop(['Prediction'], 1))  # drop Prediction column
X = preprocessing.scale(X)  # scale to normalize data
X = X[:-forecast_out]
X_forecast = X[X.shape[0] - forecast_out:]

y = np.array(df['Prediction'])
y = y[:-forecast_out]  # removes 30days of NaN

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.008)

# Training----------------------------------------------------------------------
clf = LinearRegression()
# clf = svm.SVR()
clf.fit(X_train, y_train)

# Testing---------------------------------------------------------------------------

confidence = clf.score(X_test, y_test)
result = clf.predict(X_test)
print("\nconfidence:", confidence)
# plt---------------------------------------------------------------------------
plt.figure()
plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
plt.title('LR score: %f')
plt.legend()
plt.savefig('./img/LR.png')
plt.show()

forecast_prediction = clf.predict(X_forecast)
print("\nforecast_prediction:\n", forecast_prediction)
