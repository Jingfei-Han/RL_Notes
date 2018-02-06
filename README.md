# RL_Notes
记录Richard S. Sutton的Reinforcement learning: an introduction的学习笔记

本书首先对RL做了brief review. 后面的章节共分成3大部分：tabular solution methods, 
Approximate solution methods, looking deeper。

+ 表格方法（第2~8章）
+ 近似方法（第9~13章）
+ 更深入学习（第14~18章）

其中表格方法表示存在明确解的问题。状态和动作可以用表格的形式表示

近似方法是指使用函数近似(Function Approximation)来近似表示Q值等。

更深入学习是指强化学习与心理学（psychology）、神经科学(neuroscience)的关系；
同时给出著名实例比如Atari游戏、Watson、AlphaGo等的应用

# 第一部分： Tabular Solution Methods

该部分主要目的是以最简单的形式介绍强化学习的核心概念。之所以说简单是因为状态空间和动作空间比较小，可以使用数组或表格来表示。
这种情况下，这些方法一般可以得到精确解。

该部分中， 第2章是多臂赌博机(Multi-armed Bandits)，这是强化学习中的一个非常特殊的问题，因为它只有一种状态。

然后第3章介绍了有限马尔科夫决策过程(Finite Markov Decision Processes)，这是一个通用的问题定义方法，主要观点是Bellman等式以及价值函数。

接下来的三章分别介绍了三种解决FMDP问题的基本方法：动态规划(Dynamic Programming)、蒙特卡洛方法(Monte Carlo Methods)、时序差分方法(Temporal-Difference Learning)。
这三种方法各有千秋：DP方法有良好的数学推导，但是必须要有一个完整的、精确的环境模型；MC方法不要求有模型，并且思路简单，但是不适合叠加计算
(Step-by-step incremental computation)；TD方法不要求模型，也可以叠加计算，但是难以分析，同时在收敛速度和高效性上也不如另外两种方法。

因此该部分的最后两章旨在描述如何联合这三类方法。首先的n-step Bootstrapping这一章描述了如何通过资格痕迹(Eligibility traces)来联合
MC方法和TD方法；最后一章的Planning and Learning with Tabular Methods探讨了如何联合TD方法和planning方法（比如DP）。