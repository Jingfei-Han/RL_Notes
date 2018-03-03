from model import constant_MC, TD
from environment import MRP
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    mrp = MRP()
    res_td = []
    for i in range(100):
        #record_rms = constant_MC(mrp, 0.03)
        record_rms = TD(mrp, 0.05)
        res_td.append(record_rms)

    res_mc = []
    for i in range(100):
        record_rms = constant_MC(mrp, 0.03)
        #record_rms = TD(mrp, 0.1)
        res_mc.append(record_rms)

    res_td = list(np.mean(np.array(res_td), axis=0))
    res_mc = list(np.mean(np.array(res_mc), axis=0))
    #print(record_rms)
    plt.plot(res_td, ".", label="TD")
    plt.plot(res_mc, "-", label="MC")
    plt.legend(loc="best")
    plt.show()