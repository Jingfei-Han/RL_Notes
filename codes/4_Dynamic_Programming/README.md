运行run.py

其中

>`pi = PolicyIteration(conf, "pi", True)`

True表示利用已经预训练好的转移和奖赏矩阵。如果为False则表示重新预训练，可能需要花费30分钟左右。

+ "pi"表示使用方法policy iteration
+ "vi"表示使用方法value iteration
