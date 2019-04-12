# -*- coding: utf-8 -*-

from data_processor import Data_loader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import technical_analysis as ta
import hmm as hmm
import utils
import chmm as chmm

def transform_data(data, obs_states):
    rsi = ta.rsi(data, period=4, smooth=3)
    result = np.zeros((rsi.values.shape[0], obs_states))
    for i in range(rsi.values.shape[0]):
        index = np.where(obs_range > rsi.values[i])[0][0] - 1
        result[i, index] = 1
    return result

def load_data():
    # load data
    folder = 'data\\'
    HSDataFile = "hs300.csv"  # quoted in CHFUSD so need to invert it
    data_HS_obj = Data_loader(
        HSDataFile, folder, True, delimiter=',', date_format='%Y%m%d %H%M%S')
    SZDataFile = "sz50.csv"
    data_SZ_obj = Data_loader(
        SZDataFile, folder, True, delimiter=',', date_format='%Y%m%d %H%M%S')


    # create combined data
    HS_dict = {
        'Date': data_HS_obj.get_field('Date'),
        'HS': data_HS_obj.get_field('Close')}  # inversion to get USDCHF from CHFUSD
    SZ_dict = {
        'Date': data_SZ_obj.get_field('Date'),
        'SZ': data_SZ_obj.get_field('Close')}
    df_HS = pd.DataFrame.from_dict(HS_dict).set_index('Date')
    df_SZ = pd.DataFrame.from_dict(SZ_dict).set_index('Date')
    df_joined = df_HS.join(df_SZ, how='inner')
    df_joined['nb'] = np.arange(df_joined.shape[0])

    # takes every 10 minutes so drop all indexes
    period = 1
    df = df_joined[df_joined.nb % period == 0]
    return df

# load data
df = load_data()
# plot data
plt.figure(figsize=(12, 5))
plt.xlabel('Dates')
ax1 = df.HS.plot(color='blue', grid=True, label='HS300')
ax2 = df.SZ.plot(color='red', grid=True, secondary_y=True, label='SZ50')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
lgd = plt.legend(h1 + h2, l1 + l2, loc=4)
plt.title('Price of HS300 and SZ50 (2010/01/01 to 2017/12/31)')
utils.save_figure('./img/',plt, 'Main', '1', lgd)

# compute data
obs_states = 8
obs_range = np.linspace(0, 100, obs_states + 1)  # 0., 0.125, 0.25, ...
hs_data = transform_data(df.HS, obs_states)
sz_data = transform_data(df.SZ, obs_states)
chmm_data = np.array([ np.ravel(np.outer(hs_data[t, :], sz_data[t, :])) for t in range(sz_data.shape[0])])
hidden_states = 5

# show some results
def get_hmm_results(data, name, hidden_states):
    obs_states = data.shape[1]
    hmm_obj = hmm.Hmm(data, 'rescaled', hidden_states, obs_states)
    hmm_obj.compute_proba()
    hmm_obj.plot_proba(200, 'Conditional proba ({})'.format(
        name), '{}'.format(name), '1')
    hmm_obj.EM(True)
    hmm_obj.print_parameters()
    hmm_obj.plot_proba(200, 'Conditional proba ({})'.format(
        name), '{}'.format(name), '2')
    hmm_obj.plot_likelihood('{}'.format(name), '3')
    hmm_obj.compute_viterbi_path()
    hmm_obj.plot_most_likely_state(
        200, 'Viterbi ({})'.format(name), '{}'.format(name), '4')
    return hmm_obj

def get_chmm_results(data, name, hidden_states):
    obs_states = data.shape[1]
    chmm_obj = chmm.CHmm(data, 'log-scale', hidden_states, obs_states)
    chmm_obj.compute_proba()
    chmm_obj.plot_proba(200, 'Conditional proba ({})'.format(
        name), '{}'.format(name), '1')
    chmm_obj.EM(True)
    chmm_obj.print_parameters()
    chmm_obj.plot_proba(200, 'Conditional proba ({})'.format(
        name), '{}'.format(name), '2')
    chmm_obj.plot_likelihood('{}'.format(name), '3')
    chmm_obj.compute_viterbi_path(data)
    chmm_obj.plot_most_likely_state(
        data, 200, 'Viterbi ({})'.format(name), '{}'.format(name), '4')
    return chmm_obj

# individual hmms
hs_hmm = get_hmm_results(hs_data, 'HS', hidden_states)
sz_hmm = get_hmm_results(sz_data, 'SZ', hidden_states)

# joint hmm
joined_hmm = get_chmm_results(chmm_data, 'HS-SZ', hidden_states * hidden_states)

