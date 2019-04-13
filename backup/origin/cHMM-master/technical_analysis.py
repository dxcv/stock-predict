# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


'''
The RSI (Relative Strength Index) is a price-following oscillator that ranges between 0 and 100.
'''


def rsi(data, period=14, smooth=3):

    # Get the difference in price from previous step
    delta = data.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the SMA
    avg_up = up.rolling(period).mean()
    avg_down = down.abs().rolling(period).mean()

    # Calculate the RSI based on SMA
    term = np.divide(avg_up, avg_down, where=avg_down != 0)
    term[term != term] = 1e6
    rsi = 100.0 - (100.0 / (1.0 + term))
    # unused rsi_avg = rsi.ewm(alpha=2.0/(1.0 + smooth)).mean()

    return rsi
