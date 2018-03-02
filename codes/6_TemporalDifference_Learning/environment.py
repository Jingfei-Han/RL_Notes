import numpy as np

class MRP(object):
    def __init__(self):
        self.states = ["L", "A", "B", "C", "D", "E", "R"]
        self.states_n = len(self.states)

    def reset_env(self):
        mid = self.states_n // 2
        return self.states[mid]

    def step(self, state):
        assert state in self.states

        cur_idx = self.states.index(state)
        if cur_idx == 0 or cur_idx == self.states_n - 1:
            done = True
            reward = 0
            next_state = state
            return next_state, reward, done
        else:
            if np.random.rand() < 0.5:
                next_state = self.states[cur_idx - 1]
            else:
                next_state = self.states[cur_idx + 1]

            if next_state == "R":
                reward = 1
                done = True
            elif next_state == "L":
                reward  = 0
                done = True
            else:
                reward = 0
                done = False
            return next_state, reward, done