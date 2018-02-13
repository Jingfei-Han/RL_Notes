#encoding:utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import poisson
import time
from utils import get_prob

class PolicyIteration(object):
    def __init__(self, conf, read_model=False):
        self.read_model = read_model
        self.all_trans_path = conf.all_trans_path
        self.all_rewards_path = conf.all_rewards_path


        self.rent_return = conf.rent_return
        self.move_cost = conf.move_cost

        #max count of cars in every company
        self.max_a = conf.max_a
        self.max_b = conf.max_b
        self.max_move = conf.max_move #晚上最多运max_move的车辆

        #MDP parameter
        self.gamma = conf.gamma

        #定义分布
        self.ps_a_rent = poisson(conf.lambda_a_rent)
        self.ps_b_rent = poisson(conf.lambda_b_rent)
        self.ps_a_return = poisson(conf.lambda_a_return)
        self.ps_b_return = poisson(conf.lambda_b_return)

        self.state_n = self.max_a * self.max_b
        self.action_n = 2 * self.max_move + 1

        #initialize
        #self.this_policy = np.zeros((self.state_n))
        self.this_policy = np.random.randint(0, self.action_n, (self.state_n))
        self.value_function = np.zeros((self.state_n))

        self.train()

        #print(self.this_policy.reshape(self.max_a, self.max_b))

    def train(self):
        if self.read_model:
            self.all_trans = np.load(self.all_trans_path)
            self.all_rewards = np.load(self.all_rewards_path)
        else:
            self.all_trans, self.all_rewards = self.preprocess()
            np.save(self.all_trans_path, self.all_trans)
            np.save(self.all_rewards_path, self.all_rewards)

        cnt = 0
        while True:
            #policy evaluation
            #当前policy的转移矩阵
            cur_trans_matrix = np.zeros((self.state_n, self.state_n))
            cur_rewards_matrix = np.zeros((self.state_n, self.state_n))
            for idx, action in enumerate(self.this_policy):
                cur_trans_matrix[idx, :] = self.all_trans[idx, action, :]
                cur_rewards_matrix[idx, :] = self.all_rewards[idx, action, :]

            while True:
                tmp_V = np.tile(self.value_function, (self.state_n, 1))
                new_V = np.sum((cur_trans_matrix*(cur_rewards_matrix + self.gamma * tmp_V)), axis=1)
                assert new_V.shape == self.value_function.shape
                delta = np.max(abs(new_V-self.value_function))
                print(delta)
                self.value_function = new_V
                if delta < 0.0000001:
                    break

            #policy improvement
            tmp_policy = np.zeros((self.action_n, self.state_n))
            tmp_V = np.tile(self.value_function, (self.state_n, 1))

            for i in range(self.action_n):
                tmp_policy[i, :] = np.sum(self.all_trans[:, i, :]*(self.all_rewards[:, i, :]+self.gamma*tmp_V), axis=1)

            tmp_policy = np.argmax(tmp_policy, axis=0)
            #print(tmp_policy.shape)
            #print(self.this_policy.shape)
            assert tmp_policy.shape == self.this_policy.shape
            if np.max(abs(tmp_policy - self.this_policy)) == 0:
                break
            else:
                self.this_policy = tmp_policy
            cnt += 1
            print("Current CNT: ", cnt)

        self.this_policy -= 5 #恢复到action
        print(self.this_policy.reshape(self.max_a, self.max_b))



    def preprocess(self):
        #用来预计算状态转移，返回概率矩阵和reward矩阵
        #state * action * next_state
        trans_matrix = np.zeros((self.max_a, self.max_b, self.action_n, self.max_a, self.max_b))
        rewards_matrix = np.zeros((self.max_a, self.max_b, self.action_n, self.max_a, self.max_b))

        #print(trans_matrix.shape)
        #print(self.state_n, self.action_n, self.state_n)
        #rewards_matrix.reshape(self.state_n, self.action_n, self.state_n)

        for state_a in range(self.max_a):
            #选择一个初始状态
            start1 = time.time()
            for state_b in range(self.max_b):
                #start2 = time.time()
                for action in range(self.action_n):
                    #选择一个动作
                    for next_state_a in range(self.max_a):
                        #选择一个下一个状态
                        for next_state_b in range(self.max_b):
                            trans_matrix[state_a, state_b, action, next_state_a, next_state_b], \
                            rewards_matrix[state_a, state_b, action, next_state_a, next_state_b] \
                                = self.compute(state_a, state_b, action-self.max_move, next_state_a, next_state_b)
                #end2 = time.time()
                #print("State B is: ", str(state_b), " Finish sub-iteration: ", end2 - start2)
            end1 = time.time()
            print("State A is: ", str(state_a), " Finish one state iteration: ", end1 - start1)

        #变换矩阵
        trans_matrix = trans_matrix.reshape(self.state_n, self.action_n, self.state_n)
        rewards_matrix = rewards_matrix.reshape(self.state_n, self.action_n, self.state_n)

        return trans_matrix, rewards_matrix


    def compute(self, state_a, state_b, action, next_state_a, next_state_b):
        #获取从状态(state_a, state_b)经过action到达(next_state_a, next_state_b)的状态转移概率和reward
        # action表示从A到B运送的货物量，范围是[-5, 5]
        if (action > 0 and state_a < action) or (action < 0 and state_b < -action):
            #不可能出现
            return 0, 0

        morn_state_a = state_a - action
        morn_state_b = state_b + action #morn表示morning，就是刚换完时候的状态
        if morn_state_a >= self.max_a or morn_state_b >= self.max_b:
            #不能超过最大值
            return 0, 0
        p, r = self.do_simulate(morn_state_a, morn_state_b, next_state_a, next_state_b)
        r -= np.absolute(action) * self.move_cost
        return p, r

    def do_simulate(self, morn_state_a, morn_state_b, next_state_a, next_state_b):
        #先考虑a, 变化全部来自于还的减去借的
        p_a = 0
        r_a = 0
        diff_a = next_state_a - morn_state_a #diff = return - rent

        a_max_rent = morn_state_a #最多借早上有的车辆数
        for rent_a in range(a_max_rent+1):
            return_a = diff_a + rent_a
            if return_a < 0:
                continue
            rent_a_prob = get_prob(self.ps_a_rent, rent_a)
            return_a_prob = get_prob(self.ps_a_return, return_a)

            tmp_prob = rent_a_prob * return_a_prob
            p_a += tmp_prob
            r_a += tmp_prob * rent_a * self.rent_return

        p_b = 0
        r_b = 0
        diff_b = next_state_b - morn_state_b #diff = return - rent
        b_max_rent = morn_state_b
        for rent_b in range(b_max_rent+1):
            return_b = diff_b + rent_b
            if return_b < 0:
                continue
            rent_b_prob = get_prob(self.ps_b_rent, rent_b)
            return_b_prob = get_prob(self.ps_b_return, return_b)

            tmp_prob = rent_b_prob * return_b_prob
            p_b += tmp_prob
            r_b += tmp_prob * rent_b * self.rent_return

        return p_a*p_b, r_a+r_b
