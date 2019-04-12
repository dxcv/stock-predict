# -*- coding: utf-8 -*-
# BACKTEST Module
# 回测模块
import hmm
import matplotlib.pyplot as plt
import numpy as np


# 统一化状态
# 将所有状态划为三类:涨,跌,平
# @state 1: up
# @state -1: down
# @state 0: flat
def standard_states(origin_series, up_list):
    standard_seris = []
    for i in origin_series:
        if i in up_list:
            standard_seris.append(1)
        else:
            standard_seris.append(-1)
    return standard_seris


# @deal_money 成交金额
# @cr commission rate 手续费率
def get_commission(deal_money, cr):
    commission = deal_money * cr
    if commission < 5:
        commission = 5
    return commission


# 计算: 可买入股数
# @ac 可用资金
# @price 买入价格
# @cr(commission rate) 手续费率
def get_available_lots(ac, price, cr):
    available = int(ac / price / 100)  # 计算可买入手数
    if available < 1:  # 可买手数小于1手
        return 0
    commission_money = get_commission(available * 100 * price)  # 手续费
    money = available * 100 * price + commission_money  # 成交总金额
    if money <= ac:  # 计入手续费金额充足
        return available
    else:  # 计入手续费金额不足
        for a in range(available, 0, -1):
            if a * 100 * price + get_commission(a * 100 * price) <= ac:
                return ac
        return 0


# 计算: 成交金额&成交手续费
# lots: 成交手数
# price: 成交均价
# c: commission 手续费
# @return: 成交总金额, 成交手续费
def get_deal_money(lots, price, cr):
    return price * lots * 100 * (1 + cr), price * lots * 100 * cr


# 计算最大回撤
def get_max_drawback(cum_return):
    cum_return = np.array(cum_return)
    cum_return += 1
    pre_high = 1.0  # 前高点初始化为0
    max_drawdown = 0.0  # 最大回撤
    for i in range(0, len(cum_return)):
        if cum_return[i] > pre_high:
            pre_high = cum_return[i]
        if cum_return[i] < pre_high:
            if 1 - cum_return[i] / pre_high > max_drawdown:
                max_drawdown = 1 - cum_return[i] / pre_high
    return max_drawdown


# 计算每日的累积收益率曲线
def draw_cumulative_return(cum_return):
    plt.plot(cum_return)
    plt.show()


# 回测函数
# predict_series: 状态序列(1,0,-1)
# origin_df: 每日行情数据
def backtest(predict_series, selected_states, origin_df, init_money=10000000.0):
    commission_rate = 0  # 单边1.5‰
    position = [0]  # 仓位[0,1](盘后结算)
    total_capital = [init_money]  # 总资产(盘后结算)
    available_capital = [init_money]  # 可用资金10w(交易结算)
    commission_money = 0  # 手续费(交易结算)
    # market_shares = [0]       # 持仓股数(交易结算)
    cost_price = 0  # 开仓成本价
    cost = 0  # 开仓总成本(开仓市值+手续费)
    cost_market_value = 0  # 开仓市值
    buy_trade_num = 0  # 买入次数(交易结算)
    sell_trade_num = 0  # 卖出次数(交易结算)
    win_trade_num = 0.0  # 取胜次数(盈利的交易次数)
    is_holding = False  # 是否持仓
    cum_return = [0.0]  # 累积收益率
    position_days = 0  # 持仓天数
    if len(predict_series) != len(origin_df):
        print(len(predict_series))
        print(len(origin_df))

    for i in range(0, len(predict_series)):  # 只允许做多
        daily_open = origin_df.open[i]  # 当天开盘价
        daily_close = origin_df.close[i]  # 当日收盘价
        if predict_series[i] in selected_states:  # 买入信号
            if not is_holding:  # 当前空仓
                print('Open Long:' + str(origin_df.index[i]))  # 打印买入信号
                # availble_lots = get_available_lots(availble_capital[-1], daily_open, commission_rate)    # 可买手数
                # deal_money, commission = get_deal_money(availble_lots, daily_open, commission_rate)  # 成交金额,手续费
                # market_shares = availble_lots             # 持仓股数
                commission = available_capital[-1] * commission_rate  # 计算成交手续费
                commission_money += commission  # 更新交易费用
                cost_market_value = available_capital[-1] - commission  # 持仓成本市值
                cost = available_capital[-1]  # 买入总成本(市值+手续费)
                # availble_capital.append(availble_capital[-1]-deal_money)              # 更新可用资金
                buy_trade_num += 1  # 更新交易次数
                cost_price = daily_open  # 当天开盘价作为成本价
                position_start_date = origin_df.index[i]  # 记录开仓日期
                print(u'买入: ' + str(position_start_date) + u'   成交费用: ' + str(commission) + u'元')
                is_holding = True
            if is_holding:  # 当前有持仓，继续持有d
                position_days += 1  # 更新持仓天数
            total_capital.append(daily_close / cost_price * cost_market_value)
            cum_return.append((daily_close / cost_price * cost_market_value - available_capital[0]) / available_capital[0])
            available_capital.append(0.0)  # 持仓时可用资金为0
            position.append(1)
        else:  # 卖出信号
            if is_holding:  # 当前持仓
                print('Close Long:' + str(origin_df.index[i]))  # 打印卖出平仓信号
                sell_trade_num += 1  # 更新卖出次数
                commission = daily_open / cost_price * cost_market_value * commission_rate
                commission_money += commission  # 更新交易费用
                if (daily_open / cost_price * cost_market_value - commission) / cost > 1:  # 去除手续费后仍盈利
                    win_trade_num += 1
                available_capital.append(daily_open / cost_price * cost_market_value - commission)
                position_end_date = origin_df.index[i]  # 记录平仓日期
                print(u'卖出: ' + str(position_end_date) + u'   持仓: ' + str(position_days) + u'天' + \
                      u'   成交费用: ' + str(commission) + u'元')
                position_days = 0
                is_holding = False
                cum_return.append((available_capital[-1] - available_capital[0]) / available_capital[0])
                total_capital.append(available_capital[-1])
            else:  # 空仓
                cum_return.append(cum_return[-1])  # 累积收益维持不变
                available_capital.append(available_capital[-1])
                total_capital.append(total_capital[-1])
            position.append(0)
    draw_cumulative_return(cum_return)
    print(u'总交易费用: ' + str(commission_money) + u'元')
    print(u'策略收益率: ' + str(cum_return[-1] * 100) + u'%')
    print(u'基准收益率: ' + \
          str((np.array(origin_df.close)[-1] - np.array(origin_df.close)[0]) / np.array(origin_df.close)[0] * 100) + '%')
    print(u'策略年化收益率: ' + str((cum_return[-1]) / len(origin_df) * 250 * 100) + u'%')
    print(u'基准年化收益率: ' + \
          str(((np.array(origin_df.close)[-1] - np.array(origin_df.close)[0]) / np.array(origin_df.close)[0]) /
              len(origin_df) * 250 * 100) + u'%')
    print(u'买入次数: ' + str(buy_trade_num))
    print(u'卖出次数: ' + str(sell_trade_num))
    if sell_trade_num == 0:
        print(u'胜率: --')
    else:
        print(u'胜率: ' + str(win_trade_num / sell_trade_num * 100) + u'%')
    print(u'最大回撤: ' + str(get_max_drawback(np.array(cum_return)) * 100) + u'%')

    print(u'夏普比率: ' + str(((cum_return[-1]) / len(origin_df) * 250) / np.array(cum_return).std() * np.sqrt(250)))
    print(u'夏普比率: ' + str(np.array(cum_return).mean() / np.array(cum_return).std()))
