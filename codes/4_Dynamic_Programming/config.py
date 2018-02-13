#encoding:utf-8

#记录一些具体参数，假设两个公司分别是a, b
class Config(object):
    #reward
    rent_return = 10
    move_cost = 2

    #poisson parameters
    lambda_a_rent = 3
    lambda_b_rent = 4
    lambda_a_return = 3
    lambda_b_return = 2

    #max count of cars in every company
    max_a = 21
    max_b = 21
    max_move = 5 #晚上最多运max_move的车辆

    #MDP parameter
    gamma = 0.9

    all_trans_path = "data/trains.npy"
    all_rewards_path = "data/rewards.npy"

