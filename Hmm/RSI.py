import numpy as np
import math
import random
import json
import os
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
'''
succeeded 4/11/2019
'''


# 从本地读取历史数据，下面读取的是已经下载到本地的CSV文件。
def get_stock_hist(num):
    s_his = np.genfromtxt('./data/000016.csv'.format(num), delimiter=',')
    s_hi = s_his[1:][:]
    days = s_hi.shape[0]
    this_stock = []
    for i in range(1, days, 1):
        this_day = [i]
        for k in range(1, 7):
            this_day.append(s_hi[i][k])
        this_stock.append(this_day)
    print('Maximum date is ', len(this_stock))
    return this_stock


# 均值
def get_ma(D, N):
    p_used = np.zeros(N);
    for i in range(1, N + 1, 1):
        p_used[i - 1] = stock_hist[(D - 1) - (i - 1)][4]
    ma = np.mean(p_used)
    return ma


# 获取第D天的某个数据
def get_price(D, p_tpe):
    if p_tpe == 'close':
        pos = 4;
    elif p_tpe == 'open':
        pos = 1;
    elif p_tpe == 'high':
        pos = 2;
    elif p_tpe == 'low':
        pos = 3;
    else:
        pos = 5
    price = stock_hist[D - 1][pos];
    return price


# 跨度设置为N天时，得到第D天的RSI。这里使用了简单移动平均法计算。
def get_RSI(D, N):
    up_value = 0.0
    down_value = 0.0
    for i in range(N):
        value = get_price(D - i, 'close') - get_price(D - i - 1, 'close')
        if value >= 0:
            up_value += value
        else:
            down_value -= value
    RSI = 100.0 * up_value / (up_value + down_value)
    return RSI


# Date\Open\High\Low\Close
def get_tuples(fro, to):
    res = []
    for d in range(fro, to + 1):
        tmp = []
        tmp.append(d)
        tmp.append(get_price(d, 'open'))
        tmp.append(get_price(d, 'high'))
        tmp.append(get_price(d, 'low'))
        tmp.append(get_price(d, 'close'))
        res.append(tmp)
    return res


def get_mar(fro, to, N):
    ma = []
    for i in range(fro, to + 1):
        ma.append(get_ma(i, N))
    return ma


def get_rsi(fro, to, N):
    res = []
    for d in range(fro, to + 1):
        res.append(get_RSI(d, N))
    return res


# 绘图函数
def plot_RSI(fro, to):
    rsi14 = get_rsi(fro, to, 14)
    rsi7 = get_rsi(fro, to, 7)
    ma5 = get_mar(fro, to, 5)
    ma10 = get_mar(fro, to, 10)
    ma20 = get_mar(fro, to, 20)
    tuples = get_tuples(fro, to)
    date = [d for d in range(fro, to + 1)]

    fig = plt.figure(figsize=(8, 5), dpi=600)
    p1 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4, facecolor='k')
    p1.set_title("Relative Strength Index(RSI)")
    p1.set_ylabel("Price")
    p1.plot(date, ma5, 'm')
    p1.plot(date, ma10, 'b')
    p1.plot(date, ma20, 'y')
    p1.legend(('MA5', 'MA10', 'MA20'))
    p1.grid(True, color='w')
    candlestick_ohlc(p1, tuples, width=0.7, colorup='r', colordown="g")

    p2 = plt.subplot2grid((4, 4), (3, 0), colspan=4, facecolor='g')
    p2.set_ylabel("RSI")
    p2.set_xlabel("Dates")
    p2.plot(date, rsi14, 'c-')
    p2.plot(date, rsi7, 'r-')
    p2.axhline(75, color='yellow')
    p2.axhline(25, color='yellow')
    p2.set_ylim(0, 100)
    p2.set_yticks([25, 50, 75])
    p2.legend(('RSI14', 'RSI7'), loc='upper left')
    plt.subplots_adjust(hspace=0)

    filename = './img/RSI.png'
    plt.savefig(filename, dpi=600)
    plt.show()  # show the plot on the screen

# 这里我们随意挑选一只股票测试
stock_hist = get_stock_hist(1)
# 选择绘制日期起始，例如第100天到第900天之间，调用函数
plot_RSI(100, 900)
