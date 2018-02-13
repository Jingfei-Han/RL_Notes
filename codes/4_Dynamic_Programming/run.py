from config import Config
from policy_iteration import PolicyIteration

if __name__ == "__main__":
    conf = Config()
    pi = PolicyIteration(conf, "pi", True)