from environment import MRP
import numpy as np

def compute_rms(a, b):
    #a, b is dict
    assert len(a) == len(b)
    rms = []
    for state in a.keys():
        if state not in ["L", "R"]:
            rms.append(pow(a[state] - b[state], 2))

    return np.sqrt(np.sum(rms)) / 5


def constant_MC(mrp, alpha=0.1):
    #first-visit MC method
    # return: estimated value of all states
    states = ["A", "B", "C", "D", "E"]
    true_values = {state: (i+1)*1.0/6 for i,state in enumerate(states)}
    values = {state : 0.5 for state in states}

    #update rule: new = old + alpha * (target - old)
    record_rms = []
    for episode in range(100):
        state = mrp.reset_env()
        reward_list = []
        state_list = [state]
        done = False
        while not done:
            state, reward, done = mrp.step(state)
            reward_list.append(reward)
            state_list.append(state)
        reward = reward_list[-1]
        for state in states:
            if state in state_list:
                values[state] = values[state] + alpha * (reward - values[state])
        """
        for state in state_list:
            if state in states:
                values[state] = values[state] + alpha * (reward - values[state])
        """

        rms = compute_rms(values, true_values)
        record_rms.append(rms)
    return record_rms

def TD(mrp, alpha=0.1):
    #first-visit MC method
    # return: estimated value of all states
    states = ["A", "B", "C", "D", "E", "L","R"]
    true_values = {state: (i+1)*1.0/6 for i,state in enumerate(states)}
    values = {state : 0.5 for state in states}

    true_values["L"] = 0
    true_values["R"] = 0
    values["L"] = 0
    values["R"] = 0

    #update rule: new = old + alpha * (target - old)
    record_rms = []
    for episode in range(100):
        state = mrp.reset_env()
        done = False
        while not done:
            next_state, reward, done = mrp.step(state)
            values[state] = values[state] + alpha * (values[next_state] + reward - values[state])
            state = next_state

        rms = compute_rms(values, true_values)
        record_rms.append(rms)
    return record_rms
