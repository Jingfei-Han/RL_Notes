#encoding:utf-8

#记录一些需要用到的函数
import numpy as np
from scipy.stats import poisson

def get_prob(ps, k):
    # ps是一个定义好的泊松分布
    return ps.pmf(k)

