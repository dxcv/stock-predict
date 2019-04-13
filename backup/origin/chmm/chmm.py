from __future__ import print_function

import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM
import pdb
import itertools
import tushare as ts
import matplotlib.pyplot as plt
import backtest
import sys
from hmmlearn.hmm import GaussianHMM, GMMHMM
import seaborn as sns
import numpy as np
import pandas as pd
import features
from sklearn.externals import joblib
import dao
import datetime

###############################################################################
# Downloading the data
date1 = datetime.date(2008, 1, 1)  # start date
date2 = datetime.date(2014, 1, 14)  # end date
# get quotes from yahoo finance

quotes1_train = ts.get_k_data('hs300',start='2010-01-01',end='2014-12-31')
quotes2_train = ts.get_k_data('sz50',start='2010-01-01',end='2014-12-31')
quotes1_test = ts.get_k_data('hs300',start='2015-01-01',end='2017-12-31')
quotes2_test = ts.get_k_data('sz50',start='2015-01-01',end='2017-12-31')


#if len(quotes1) == 0 or len(quotes1) != len(quotes2):
#   raise SystemExit
dates = quotes1_train['date'].tolist()
# unpack quotes
close_v1_train = np.array(quotes1_train['close'].tolist())
volume1_train = np.array(quotes1_train['volume'].tolist())
close_v2_train = np.array(quotes2_train['close'].tolist())
volume2_train = np.array(quotes2_train['volume'].tolist())

close_v1_test = np.array(quotes1_test['close'].tolist())
volume1_test = np.array(quotes1_test['volume'].tolist())
close_v2_test = np.array(quotes2_test['close'].tolist())
volume2_test = np.array(quotes2_test['volume'].tolist())

# take diff of close value
# this makes len(diff) = len(close_t) - 1
# therefore, others quantity also need to be shifted
diff1_train = close_v1_train[1:] - close_v1_train[:-1]
volume1_train = volume1_train[1:]
diff2_train = close_v2_train[1:] - close_v2_train[:-1]
volume2_train = volume2_train[1:]

diff1_test = close_v1_test[1:] - close_v1_test[:-1]
volume1_test = volume1_test[1:]
diff2_test = close_v2_test[1:] - close_v2_test[:-1]
volume2_test = volume2_test[1:]

# pack diff and volume for training
X1_train = np.column_stack([diff1_train, volume1_train])
X2_train = np.column_stack([diff2_train, volume2_train])
X1_test = np.column_stack([diff1_test, volume1_test])
X2_test = np.column_stack([diff2_test, volume2_test])
###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end='')
n_components = 5
# make an HMM instance and execute fit
model1 = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)
model2 = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)

model1.fit(X1_train)
model2.fit(X2_train)

# predict the optimal sequence of internal hidden state
hidden_states1 = model1.predict(X1_test)
hidden_states2 = model2.predict(X2_test)

print("done\n")

# calculate similarity measure
states1 = range(n_components)
states2 = list(itertools.permutations(states1))
print(states1)
print(len(states2))
sims = []
for i in range(len(states2)):
    sim = 0
    for j in range(len(hidden_states1)):
        sim += hidden_states1[j] == states2[i][hidden_states2[j]]
        #pdb.set_trace()
    sims.append(float(sim)/len(hidden_states1))

similarity = max(sims)    
print(["similarity: ", similarity])
m_ind = sims.index(similarity)
st = states2[m_ind]

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print(model1.transmat_)
print()

print("means and vars of each hidden state")
for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model1.means_[i])
    print("var = ", np.diag(model1.covars_[i]))
    print()

years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')
fig = pl.figure()
ax = fig.add_subplot(111)
colors = ['r','b','g','m','k']

for i in range(n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states1 == i)
    ax.plot_date(quotes1_test['date'][1:][idx], quotes1_test['close'][1:][idx], 'o', label="%dth hidden state" % i, color = colors[i])

ax.legend(loc=2)
plt.show()

#used_colors = ax._get_lines.color_cycle
#pdb.set_trace()
fig = pl.figure()
ax = fig.add_subplot(111)
for i in range(n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states2 == st[i])
    ax.plot_date(quotes2_test['date'][1:][idx], quotes2_test['close'][1:][idx], 'o', label="%dth hidden state" % i, color = colors[i])
ax.legend(loc=2)
plt.show()



ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()


ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)

fig.autofmt_xdate()

pl.plot(range(len(sims)), sims)
pl.show()

backtest.backtest(hidden_states1, [0,1,2], dao.get_local_bar_data('sz50', '2015-01-01', '2017-12-31'))
