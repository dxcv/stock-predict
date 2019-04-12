# -*- coding: utf-8 -*-
# 模型训练
# Build and TRAIN HMM models
# @Data Source: TuShare
# @Model: Hidden Markov model
import sys
import os
from hmmlearn.hmm import GaussianHMM, GMMHMM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import features
from sklearn.externals import joblib
import dao
import datetime
import shutil
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 训练集每日状态散点图
# @update: 12.5 支持14种状态不同颜色显示
def draw_result(save_path,dates, closes, lss, model):
    sns.set_style('white')
    plt.figure(figsize=(8, 5),dpi=600)
    plt.xlabel('Year')
    plt.ylabel('S&P 500 Index')
    # dates = dates[1:]
    # closes = closes[1:]
    for i in range(model.n_components):
        state = (lss == i)[:-1]
        if i == 13:
            plt.plot(dates[state], closes[state], '.', color='#404080', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 12:
            plt.plot(dates[state], closes[state], '.', color='#408040', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 11:
            plt.plot(dates[state], closes[state], '.', color='#804040', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 10:
            plt.plot(dates[state], closes[state], '.', color='#80FF80', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 9:
            plt.plot(dates[state], closes[state], '.', color='#8080FF', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 8:
            plt.plot(dates[state], closes[state], '.', color='#FF8080', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 7:
            plt.plot(dates[state], closes[state], '.', color='#808080', label='latent state %d' % i, lw=1,
                     markersize=10)
        elif i == 6:
            plt.plot(dates[state], closes[state], '.', color='k', label='latent state %d' % i, lw=1,
                     markersize=10)
        else:
            plt.plot(dates[state], closes[state], '.', label='latent state %d' % i, lw=1, markersize=10)
        plt.legend(fontsize=14, loc=2)
        plt.grid(1)
    filename = save_path+'/S&P_500_Index.png'
    count = 1
    while os.path.isfile(filename):
        count = count + 1
        filename = save_path+'/S&P_500_Index({}).png'.format(count)
    plt.savefig(filename, dpi=600)
    plt.show()


# 训练集模拟收益率曲线
# @update: 12.5 支持14种状态不同颜色显示
def draw_back(save_path,dates, lr, lss, model):
    data = pd.DataFrame({'datelist': dates, 'logreturn': lr, 'state': lss[:-1]}).set_index(
        'datelist')
    plt.figure(figsize=(8, 5),dpi=600)
    plt.xlabel('Year')
    plt.ylabel('Net Value')
    candidate_state_set = []  # 候选状态集
    for i in range(model.n_components):
        state = (lss == i)[:-1]
        # idx = np.append(0, state)
        idx = state
        data['state %d_return' % i] = data.logreturn.multiply(idx, axis=0)
        cum_return = np.exp(data['state %d_return' % i].cumsum())
        # 筛选候选状态
        if cum_return[-1] > 1.15:  # 累积收益率大于20%
            candidate_state_set.append(i)
        elif cum_return[-1] > 1.05 and True not in (cum_return < 0.97).tolist():  # Or:累积收益>0且最大亏损不超过10%
            candidate_state_set.append(i)
        # elif np.sum(cum_return > 1) > len(cum_return)*0.7 and np.sum(cum_return < 1) < len(cum_return)*0.05:
        #    candidate_state_set.append(i)
        if i == 13:
            plt.plot(cum_return, color='#404080', label='latent_state %d' % i, markersize=10)
        elif i == 12:
            plt.plot(cum_return, color='#408040', label='latent_state %d' % i, markersize=10)
        elif i == 11:
            plt.plot(cum_return, color='#804040', label='latent_state %d' % i, markersize=10)
        elif i == 10:
            plt.plot(cum_return, color='#80FF80', label='latent_state %d' % i, markersize=10)
        elif i == 9:
            plt.plot(cum_return, color='#8080FF', label='latent_state %d' % i, markersize=10)
        elif i == 8:
            plt.plot(cum_return, color='#FF8080', label='latent_state %d' % i, markersize=10)
        elif i == 7:
            plt.plot(cum_return, color='#808080', label='latent_state %d' % i, markersize=10)
        elif i == 6:
            plt.plot(cum_return, color='k', label='latent_state %d' % i, markersize=10)
        else:
            plt.plot(cum_return, label='latent_state %d' % i, markersize=10)
        plt.legend(fontsize=14, loc=2)
        plt.grid(1)
    filename = save_path+'/Net_Value.png'
    count = 1
    while os.path.isfile(filename):
        count = count + 1
        filename = save_path +'/Net_Value({}).png'.format(count)
    plt.savefig(filename, dpi=600)
    plt.show()
    print('states:' + str(candidate_state_set) + ' selected.')
    return candidate_state_set


# 动态跟踪每一个状态的盈利情况
# @param lss: 状态序列
#        lr: 对数收益率
#        states_num: 状态数
# @return candidate_state_set 候选状态集
def state_dynamic_trace(lr, lss, states_num):
    data = pd.DataFrame({'logreturn': lr, 'state': lss[:-1]})
    candidate_state_set = []
    for i in range(0, states_num):  # 统计每个状态的收益情况
        state = (lss == i)[:-1]
        idx = state
        data['state %d_return' % i] = data.logreturn.multiply(idx, axis=0)
        cum_return = np.array(np.exp(data['state %d_return' % i].cumsum()))
        # 设置状态累积收益条件
        # 条件1：累积收益>15%
        if cum_return[-1] > 1.15:
            candidate_state_set.append(i)
        # 条件2：累积收益>5%且最大亏损<3%
        elif cum_return[-1] > 1.05 and True not in (cum_return < 0.97).tolist():
            candidate_state_set.append(i)
    return candidate_state_set


# 一次性生成所有交易日的属性,但每个迭代只预测一个交易日的状态
# @return: lss 状态序列
def simulate_predict(X, hmm):
    lss = []
    for i in range(1, len(X) + 1):
        temp_x = X[0:i]
        lss.append(hmm.predict(temp_x)[-1])
    lss = np.array(lss)
    return lss


def main():
    ''' 标的 '''
    code = '000300'
    ''' 个股 '''
    code = '000016'
    img_path = './img/'+code
    train_s_date = '2005-01-01'  # 训练集时间范围
    train_e_date = '2010-12-31'
    test_s_date = '2011-01-01'  # 测试集时间范围
    test_e_date = '2017-12-31'
    # test_e_date = datetime.datetime.now().strftime('%Y-%m-%d')
    shutil.rmtree(img_path)
    os.mkdir(img_path)
    is_feature_select = False  # 是否特征选择
    is_standardize = False  # 是否标准化
    maxperiod = 33  # 生成属性的最大周期

    index_hist_train_df = dao.get_local_bar_data(code, train_s_date, train_e_date)  # 获取训练集
    index_hist_test_df = dao.get_local_bar_data(code, test_s_date, test_e_date)  # 获取测试集
    index_hist_train_df.sort_index(inplace=True)  # 按照日期排序
    index_hist_test_df.sort_index(inplace=True)  # 按照日期排序
    train_datelist = pd.to_datetime(index_hist_train_df.close.index[maxperiod + 1:])  # 获取训练集的日期列表
    train_closeidx = index_hist_train_df.close[maxperiod + 1:]  # 获取训练集每日收盘价
    test_datelist = pd.to_datetime(index_hist_test_df.close.index[maxperiod + 1:])  # 获取测试集的日期列表
    test_closeidx = index_hist_test_df.close[maxperiod + 1:]  # 获取测试集每日收盘价
    # train
    train_X, train_logreturn = features.gen_attrs(img_path,index_hist_train_df, maxperiod)  # 生成属性
    # test
    test_X, test_logreturn = features.gen_attrs(img_path,index_hist_test_df, maxperiod)  # 生成属性
    # train hmm model
    hmm = GaussianHMM(n_components=5, covariance_type='diag', n_iter=1000000, algorithm='viterbi', random_state=0)
    hmm = hmm.fit(train_X)  # 训练HMM
    latent_states_sequence_train = simulate_predict(train_X, hmm)  # 训练集模拟每天预测一天
    draw_result(img_path,train_datelist, train_closeidx, latent_states_sequence_train, hmm)  # 训练集状态图
    draw_back(img_path,train_datelist, train_logreturn, latent_states_sequence_train, hmm)  # 训练集收益曲线
    latent_states_sequence_test = simulate_predict(test_X, hmm)  # 测试集模拟每天预测一天
    latent_states_sequence_test[:-1].tofile('states_sequence.csv', sep=',')  # 保存模拟预测每日状态
    draw_result(img_path,test_datelist, test_closeidx, latent_states_sequence_test, hmm)  # 测试集状态图
    selected_states = draw_back(img_path,test_datelist, test_logreturn, latent_states_sequence_test, hmm)  # 测试集收益曲线
    print(selected_states)
    joblib.dump(hmm, 'model/hmm_50.m')  # 本地化模型

    # 本地回测
    latent_states_sequence_backtest = latent_states_sequence_test[len(latent_states_sequence_train):]
    import backtest
    backtest.backtest(latent_states_sequence_backtest, [0, 1, 2], dao.get_local_bar_data(code, '2015-01-01', '2017-12-31'))
    # joblib.dump(min_max_scaler, 'model/std.m')    # 数据标准化模型
    features.gen_attrs(img_path,dao.get_local_bar_data(code, '2010-01-01', '2017-12-31'), 10)


def get_train_attrs(code, train_s_date, train_e_date, is_index, maxperiod):
    train_hist_df = dao.get_local_bar_data(code, train_s_date, train_e_date)
    train_hist_df.sort_index(inplace=True)
    train_X, logreturn = features.gen_attrs(train_hist_df, maxperiod)
    return train_X


if __name__ == "__main__":
    main()


