# -*- coding: utf-8 -*-
# dynamic state selection
# 用于市场做多状态的动态选择
import math
import talib
import numpy as np


class StateSelection:
    state_num = 0   # 状态数
    state_daily_return = {}             # 状态每日对数回报率
    state_daily_cumulative_return = {}  # 状态每日累积对数回报率
    days = 0    # 记录累积天数
    states_pool = []
    static_states_pool = []

    def __init__(self, sn, sp):
        self.state_num = sn
        self.states_pool = sp
        self.static_states_pool = sp
        for i in range(0, sn):
            self.state_daily_return[i] = [0.0]
            self.state_daily_cumulative_return[i] = [0.0]

    # 计算各个状态的单日收益率
    def update_return(self, yesterday_price, today_price, state):
        self.days += 1
        logreturn = math.log(today_price / yesterday_price)
        for i in range(0, self.state_num):
            if i == state:
                self.state_daily_return[state].append(logreturn)
                self.state_daily_cumulative_return[state].append(
                    self.state_daily_cumulative_return[state][-1] + logreturn)
            else:
                self.state_daily_return[i].append(0.0)
                self.state_daily_cumulative_return[i].append(
                    self.state_daily_cumulative_return[i][-1])

    def get_daily_return(self):
        return self.state_daily_return

    def get_state_daily_return(self, state):
        return self.state_daily_return[state]

    def get_daily_cumulative_return(self):
        return self.state_daily_cumulative_return

    def get_state_daily_cumulative_return(self, state):
        return self.state_daily_cumulative_return[state]

    def get_cumulative_return(self):
        cr = []
        for i in range(0, self.state_num):
            cr.append(self.state_daily_cumulative_return[i][-1])
        return cr

    def get_state_cumulative_return(self, state):
        return self.state_daily_cumulative_return[state][-1]

    # @return direction: 1 for up trend, -1 for down trend, 0 for unknown
    def get_state_cumulative_return_ma_direction(self, state, period):
        MA = talib.MA(np.array(self.get_state_daily_cumulative_return(state)), timeperiod=period)
        if len(MA) < 2 or len(MA) < period + 2:     # 初始阶段无法计算MA
            return 0
        else:
            if MA[-1] - MA[-2] >= 0.01:
                return 1
            elif MA[-1] - MA[-2] <= -0.01:
                return -1
            else:
                return 0

    def add_state(self, s):
        if s not in self.states_pool:
            self.states_pool.append(s)

    def remove_state(self, s):
        if s in self.states_pool:
            self.states_pool.remove(s)

    def reset_state(self, new_states_pool):
        diff_set = list(set(new_states_pool).difference(set(self.static_states_pool)))
        self.states_pool = self.static_states_pool + diff_set

    def is_state_in_pool(self, s):
        if s in self.states_pool:
            return True
        else:
            return False
