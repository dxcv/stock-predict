import math

import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.stats import multinomial


COLORS = ["r", "orange", "b", 'g', 'darkgreen', 'red', 'yellow', 'yellowgreen','gold',
          'lightskyblue','lightcoral', 'c', 'm', 'y', 'k', 'w',"r", "orange", "b", 'g', 'darkgreen', 'red',
          'yellow', 'yellowgreen','gold']

"""
coupled Hidden Markov Model class
assuming multinomial probabilities
"""


class CHmm(object):
    def __init__(self, data, hmm_type='log-scale', hidden_states=5, obs_states=8):

        if hmm_type != 'log-scale' and hmm_type != 'rescaled':
            raise RuntimeError("unknow type! allowed log-scale or rescaled")

        # the initial probability distribution
        self.K = hidden_states  # hidden states

        # number of coupled HMM (in the case of two HMMs, C=2)
        self.C = data.shape[1]
        self.obs_states = obs_states
        self.states = range(self.K)
        self.pi = np.full((self.K), 1 / self.K)
        #self.pi = np.full((self.C, self.K), 1 / self.K)

        # the probability transition matrix 转移概率矩阵
        # stays 0.5 in its current state and 0.5/(self.K-1) otherwise

        self.a = np.matrix([[abs(math.sin(i)) if i == j else (1 - abs(math.sin(i))) / (self.K - 1)
                             for j in range(self.K)] for i in range(self.K)])
        '''
        self.a_simple = np.matrix([[0.5 if i == j else 0.5 / (self.K - 1)
                                    for j in range(self.K)] for i in range(self.K)])
        self.a = [self.a_simple.copy() for c in range(self.C * self.C)]
        '''
        # the emission probabilities are multinomials发散概率多项式
        self.eta = np.array([[1 / self.obs_states for j in range(self.obs_states)]
                              for i in range(self.K)])
        '''
        self.eta = np.array([[[1 / self.obs_states for j in range(self.obs_states)]
                              for i in range(self.K)] for c in range(self.C)])
        '''
        # the coupling parameters
        self.theta = np.full((self.C, self.C), 1 / self.C)

        # store the initial value
        self.a0 = self.a
        self.pi0 = self.pi
        self.eta0 = self.eta
        self.theta0 = self.eta

        self.data = data
        self.T = data.shape[0]
        self.train_data = data   #指定时间
        self.test_data =data
        self.hmm_type = hmm_type
        return

    # compute the emission probabilities发射概率
    # based on gaussian assumptions
    def __compute_B(self, data):
        self.multinomial = [multinomial(1, self.eta[i, :]) for i in range(self.K)]
        self.b = np.zeros((self.T, self.K))
        for t in range(self.T):
            self.b[t, :] = [self.eta[y, int(data[t, y])] for y in range(self.K)]
        '''
        T = len(data)
        self.b = np.zeros((T, self.K))
        for t in range(T):
            print(data[t,:])
            self.b[t, :] = [self.eta[y, data[t, :]] for y in range(self.K)]
        '''

        # other computation for log-scale
        if self.hmm_type == 'log-scale':
            self.log_b = np.zeros((self.T, self.K))
            for t in range(self.T):
                self.log_b[t, :] = np.log(self.b[t, :])
        return

    # alpha recursion (this is a forward recursion) a递推
    # implemented either in log-scale or rescaled version
    def __alpha_recursion(self):
        if self.hmm_type == 'log-scale':
            self.log_alpha = np.zeros((self.T, self.K))
            # Initialize base cases (t == 0)
            self.log_alpha[0, :] = np.log(self.pi * self.b[0, :])

            # Run Forward algorithm for t > 0
            for t in range(self.T - 1):
                log_alpha_star = np.amax(self.log_alpha[t, :])
                for q in self.states:
                    self.log_alpha[t + 1, q] = self.log_b[t + 1, q] + log_alpha_star + \
                        math.log(sum((math.exp(self.log_alpha[t, q0] -
                                               log_alpha_star) * self.a[q0, q])for q0 in self.states))

        # rescaled case调整过的例子
        elif self.hmm_type == 'rescaled':
            self.alpha = np.zeros((self.T, self.K))

            # Initialize base cases (t == 0)
            self.alpha[0, :] = self.pi * self.b[0, :]
            s = self.alpha[0, :].sum()
            self.alpha[0, :] = self.alpha[0, :] / s

            # Run Forward algorithm for t > 0
            for t in range(self.T - 1):
                self.alpha[t + 1, :] = np.multiply(
                    self.b[t + 1, :], np.dot(self.a.T, self.alpha[t, :]))
                s = self.alpha[t + 1, :].sum()
                self.alpha[t + 1, :] = self.alpha[t + 1, :] / s
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")
        return

    # beta recursion (this is a backward recursion)b递推 向后
    # implemented either in log-scale or rescaled version
    def __beta_recursion(self):
        if self.hmm_type == 'log-scale':
            self.log_beta = np.zeros((self.T, self.K))
            for q in self.states:
                self.log_beta[self.T - 1, q] = 0
            for t in reversed(range(self.T - 1)):
                log_beta_star = np.amax(self.log_beta[t + 1, :])
                for q in self.states:
                    self.log_beta[t, q] = log_beta_star + math.log(sum((math.exp(
                        self.log_beta[t + 1, q1] - log_beta_star) * self.a[q, q1] * self.b[t + 1, q1]) for q1 in self.states)+1e-6)
        elif self.hmm_type == 'rescaled':
            self.beta = np.zeros((self.T, self.K))
            self.scale_factor_beta = np.ones(self.T)
            self.beta[self.T - 1, :] = 1 / self.K
            self.scale_factor_beta[self.T - 1] = self.K

            for t in range(self.T - 2, -1, -1):
                self.beta[t, :] = np.dot(self.a, np.multiply(
                    self.b[t + 1, :], self.beta[t + 1, :]))
                self.scale_factor_beta[t] = self.beta[t, :].sum()
                self.beta[t, :] = self.beta[t, :] / self.scale_factor_beta[t]
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")

        return

    # Q2: compute the conditional probabitilies 计算条件概率
    # based on the data
    # called the function to compute the emission probabilities 使用计算发射概率
    # as well as the alpha and beta recursion 和ab递推一样
    # implemented either in log-scale or rescaled version 用两个刻度
    def compute_proba(self, data=np.array([])):
        if data.size == 0:
            data = self.train_data
        self.__compute_B(data)
        self.__alpha_recursion()
        self.__beta_recursion()

        # initialize array
        self.cond_proba = np.zeros((self.T, self.K))
        self.joined_cond_proba = np.zeros((self.T - 1, self.K, self.K))

        # do the computation
        if self.hmm_type == 'log-scale':
            for t in range(self.T):
                amax = np.max(self.log_alpha[t, :] + self.log_beta[t, :])
                proba_sum = sum((math.exp(
                    self.log_alpha[t, zt] + self.log_beta[t, zt] - amax)) for zt in self.states)
                for q in self.states:
                    self.cond_proba[t, q] = math.exp(
                        self.log_alpha[t, q] + self.log_beta[t, q] - amax) / proba_sum
                if t < self.T - 1:
                    for q in self.states:
                        for q1 in self.states:
                            self.joined_cond_proba[t, q, q1] = math.exp(
                                self.log_alpha[t, q] + self.log_beta[t + 1, q1] - amax) \
                                * self.b[t + 1, q1] * self.a[q, q1] / proba_sum
        elif self.hmm_type == 'rescaled':
            self.cond_proba = self.alpha * self.beta
            denom = self.cond_proba.sum(1)
            self.cond_proba = self.cond_proba / denom[:, None]

            denom = denom * self.scale_factor_beta
            denom = denom[:(self.T - 1), None, None]
            self.joined_cond_proba = self.alpha[:(self.T - 1), :, None] * self.beta[1:self.T, None, :] \
                * np.asarray(self.a)[None, :, :] \
                * self.b[1:self.T, :, None].transpose((0, 2, 1)) \
                / denom
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")
        return

    # Q2: plot the conditional proba p(q_t|u) 条件概率
    def plot_proba(self, T_max, title, prefix, suffix):
        self.__plot_states(self.cond_proba, T_max, title, prefix, suffix)
        return

    def plot_most_likely_state(self, path_data, T_max, title, prefix, suffix):
        print(path_data.shape)
        data = np.zeros((len(path_data), self.K))
        for i in range(len(path_data)):
            data[i, int(path_data[i,1])] = 1 #有修改
        self.__plot_states(data, T_max, title, prefix, suffix, 'step')
        return

    def __plot_states(self, data, T_max, title, prefix, suffix, plot_type='plot'):
        f, axarr = plt.subplots(self.K,  sharex=True)
        for i in range(self.K):
            if plot_type == 'step':
                axarr[i].step(range(T_max), data[:T_max, i],
                              c=COLORS[i], label="State %d" % (i + 1))
            else:
                axarr[i].plot(range(T_max), data[:T_max, i],
                              c=COLORS[i], label="State %d" % (i + 1))
            axarr[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        f.subplots_adjust(hspace=0.2)
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(title, y=(self.K + 0.8))
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        utils.save_figure('./img/chmm/',plt, prefix, suffix, lgd)
        return

    # the incomplete lok likelihood is composed 不完全似然估计值
    # of 3 terms as explained in the pdf file
    def expected_complete_log_likelihood(self, data):
        term_1 = (self.cond_proba[0, :] * np.log(self.pi)).sum()
        term_2 = (self.joined_cond_proba * np.log(self.a[None, :, :])).sum()
        term_3 = (self.cond_proba * np.log(self.b)).sum()
        return term_1 + term_2 + term_3

    # Q3 Expectation Maximization algo 最大期望算法（Expectation Maximization Algorithm
    def EM(self, compute_likelihood=False, max_iter=200, precision=1e-4):
        if compute_likelihood:
            self.llh = []

        for i in range(max_iter):
            pi_0, a_0, eta_0 = self.pi, self.a, self.eta
            # expectation
            self.compute_proba(self.train_data)

            # maximimization
            self.pi = self.cond_proba[0, :]
            self.a = self.joined_cond_proba.sum(
                0) / self.joined_cond_proba.sum((0, 1))
            self.eta = (self.cond_proba[:, :, None] * self.train_data[:,
                                                                      None, :]).sum(0) / self.cond_proba.sum(0)[:, None]

            # compute expected_complete_log_likelihood
            if compute_likelihood:
                train_llh = self.expected_complete_log_likelihood(
                    self.train_data)
                self.compute_proba(self.test_data)
                test_llh = self.expected_complete_log_likelihood(
                    self.test_data)
                self.llh.append([train_llh, test_llh])

            # Check halt condition
            if max(np.max(np.abs(self.pi - pi_0)),
                   np.max(np.abs(self.a - a_0)),
                   np.max(np.abs(self.eta - eta_0))) < precision:
                break
            if i == max_iter:
                raise RuntimeError("max iteration reached")
        self.iterations = i + 1
        return

    # a simple helper function to print the parameters 一个简单的辅助函数打印参数
    def print_parameters(self, precision=4):
        np.set_printoptions(precision)
        print('EM converged in ', self.iterations, ' iterations')
        print('pi:', self.pi)
        print('a:', self.a)
        print('eta:', self.eta)

    # Q5 plot expected_complete_log_likelihood 完全期望对数似然图
    def plot_likelihood(self, prefix, suffix):
        plt.figure()
        self.llh = np.asarray(self.llh)
        N = len(self.llh)
        labels = ['train llh', 'test llh']
        for i in range(2):
            plt.plot(range(1, (N + 1)),
                     self.llh[:, i], c=COLORS[i], label=labels[i])
        plt.ylabel('log likelihood')
        plt.xlabel('EM iterations')
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        utils.save_figure('./img/chmm/',plt, prefix, suffix, lgd)
        return

    # Q7
    # Viterbi decoding algorithm to find the most probable path followed
    # by the latent variables Q to generate the observations u Viterbi译码算法寻找最有可能路径后潜在变量Q生成观测变量U
    def compute_viterbi_path(self, data):
        T = len(data)
        self.path = np.zeros(T)
        self.max_index = np.zeros((T, self.K))
        self.max_proba = np.zeros((T, self.K))

        if self.hmm_type == 'log-scale':
            # precompute the log
            l_pi = np.log(self.pi)
            l_a = np.log(self.a)
            l_b = np.log(self.b)
            # in log scale the max proba is the log of the max proba
            self.max_proba[0, :] = l_pi + l_b[0, :]

            # Run Viterbi for t > 0
            for t in range(1, T):
                for q in self.states:
                    (self.max_proba[t, q], self.max_index[t - 1, q]) = max(
                        (self.max_proba[t - 1, q0] + l_a[q0, q] + l_b[t, q], q0) for q0 in self.states)

            # do backward induction 做逆向归纳
            self.path[T - 1] = np.argmax(self.max_proba[T - 1, :])
            for t in range(T - 2, -1, -1):
                self.path[t] = self.max_index[t, int(self.path[t + 1])]

        elif self.hmm_type == 'rescaled':
            self.scale_factor_viterbi = np.zeros(T)

            # Initialize base cases (t == 0)
            self.max_proba[0, :] = self.pi * self.b[0, :]
            self.scale_factor_viterbi[0] = self.max_proba[0, :].sum()
            self.max_proba[0, :] = self.max_proba[0, :] / \
                self.scale_factor_viterbi[0]

            # Run Viterbi for t > 0
            for t in range(1, T):
                for q in self.states:
                    (self.max_proba[t, q], self.max_index[t - 1, q]) = max(
                        (self.max_proba[t - 1, q0] * self.a[q0, q] * self.b[t, q], q0) for q0 in self.states)
                self.scale_factor_viterbi[t] = self.max_proba[t, :].sum()
                self.max_proba[t, :] = self.max_proba[t, :] / \
                    self.scale_factor_viterbi[t]

            # do backward induction
            self.path[T - 1] = np.argmax(self.max_proba[T - 1, :])
            for t in range(T - 2, -1, -1):
                self.path[t] = self.max_index[t, int(self.path[t + 1])]
        else:
            raise RuntimeError("unknow type allowed log-scale or rescaled")
        return
