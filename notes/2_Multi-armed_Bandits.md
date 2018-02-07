# Multi-armed Bandits

---

# 1. Evaluative or instructive feedback

强化学习区别于其他学习方法的最重要一点就是：**强化学习使用训练信息去评估(Evaluate)动作而不是根据正确的动作去指导(Instruct)动作**。
下面介绍两个概念评估性反馈(Evaluative feedback)和指导性反馈(Instructive feedback)。

纯的评估性反馈就是会告诉我们当前采取的动作有多好，但是不能告诉我们当前动作是不是最好的或者最坏的动作；而纯的指导性反馈完全独立于我们
当前采取的动作，只能告诉你最好的动作是什么。可见这是两种完全相反的模式：评估性反馈完全依赖于当前选取的动作，但是指导性反馈完全独立于
当前选取的动作。

这一章主要讲述多臂老虎机(K-armed bandits)的一个简单版本，这里只有一个状态，不涉及复杂的状态转移等问题，是一个非关联(Nonassociative)的设置。
这样做也是为了简化强化学习问题，更能理清楚评估性反馈与指导性反馈的重要区别是什么。在这一章结尾，作者花了一节的篇幅介绍了赌博机问题编程关联的
(Associative)情况，也就是说，需要考虑在多个情况下采取动作的情况。

# 2. 什么是多臂老虎机问题(K-armed bandit problem)

多臂老虎机问题可以这样理解：摆在你面前一个有k个摇臂的老虎机，你可以从k个摇臂中随机选择一个按下去，然后老虎机根据你选择的动作，按照一个稳定的
概率分布(Stationary probability distribution)反馈给你一定的钱。你的目标就是在玩一定次数或之间内，比如玩1000次内，取得最大的总期望。


下面我们要一点一点的分析并定义这个问题。

在多臂老虎机中，我们可以认为动作空间大小为k，即我们从k个摇臂中选择1个。当一个动作被选择之后，即我们按下了其中一个臂之后，老虎机会按照某个
稳定的概率分布随机返回给我们奖赏，虽然奖赏是随机的，但是由于是来自一个稳定的概率分布，因此期望是稳定的，我们把这个期望称为这个价值(Value)。


我们定义在时刻t选择的动作是 
![](https://latex.codecogs.com/png.latex?A_t), 对应获得的奖赏(Reward)是
![](https://latex.codecogs.com/png.latex?R_t)。因此任意动作a的价值
![](https://latex.codecogs.com/png.latex?q_*(a))就是选择动作
a获得的奖赏的期望。即：

<center>

![](https://latex.codecogs.com/png.latex?q_*(a)\doteq E[R_t|A_t=a])

</center>

假设我们知道那个臂出的钱的期望值最大，那我们只需要每次都按那个出钱最多的臂就行了，这样我们肯定会获得最好的奖赏。但是问题在于每个臂出钱的概率
分布我们是未知的。因此我们只能根据经验，来对动作的价值进行估计。假设在时刻t我们对动作a的价值的估计为
![](https://latex.codecogs.com/png.latex?Q_t(a))。我们希望这个估值能接近于
![](https://latex.codecogs.com/png.latex?q_*(a))。

这里之所以有下标t，表示我们在每一个时间都会获得新的经验，把新的经验应用上去更加准确的估计动作的价值。因此实际我们对动作价值的估计是在不断更新的。

在这个问题中，如果我们每次都选择当前估值最大的动作，即贪心(Greedy)动作，这就是利用(Exploitation)你当前对于所有动作的估值。但是只这样做是不好的，
因为可能有和估值最大的动作估值差不多的其他动作，但是它的估值还不稳定，可能它的准确价值应该高于当前最优动作。因此我们应该对这种情况进行探索(Exploration)。
这就是要去选择那些非贪心(Non-greedy)的行为，然后对这个行为进行估值。换句话说，利用的过程其实就是贪心的过程，贪图当前的短期利益；而探索的过程
其实就是不去追求短期利益，而是为了更好的让长期利益最大。但是我们对于一个动作不可能既是探索，又是利用，因此我们需要平衡探索和利用的关系。

现有很多很复杂的方法用来解决平衡问题，但是大多数都做了很强的假设，因此这里我们只考虑一些简单的平衡(Balance)方法。在多臂老虎机上可以看到清晰的形式。

# 3. 动作-价值方法(Action-value Methods)

注意我们动作的真实价值定义
![](https://latex.codecogs.com/png.latex?q_*(a))
就是动作被选择所获得的奖赏的期望。因此一个很自然的想法就是使用所有之前的实验中选择当前动作的奖赏均值来估计动作的奖赏。公式如下

<center>

![](../images/2_Multi_armed_Bandits/sample_avg.png)

</center>

根据大数定律可知，当选择当前动作的次数趋于无穷的时候，
![](https://latex.codecogs.com/png.latex?Q_t(a))可以收敛到
![](https://latex.codecogs.com/png.latex?q_*(a))。
这种方法称为采样平均法(Sample-average)。

有了对每个动作价值的估计，我们就可以选择动作了。最简单的动作选择方法就是选择估值高的动作，其实就是贪心的选择。可以用公式表示为：

<center>

![](https://latex.codecogs.com/png.latex?A_t\doteq\underset{a}{argmax}Q_t(a))

</center>

但是我们还需要exploration，因此我们可以采用一种简单的策略：大多数时候我们贪心选择动作，只有很少的时候，比如一个
小的概率
![](https://latex.codecogs.com/png.latex?\epsilon)
我们随机(randomly)选择动作。这种近似贪心的动作选择方法，我们称为
![](https://latex.codecogs.com/png.latex?\epsilon-greedy)。

# 4. 10臂老虎机实验平台(The 10-armed Testbed)

为了方便的比较实验效果，我们决定实现一个多臂老虎机平台。我们使用2000个随机生成的10臂老虎机进行实验。
对于每个老虎机，每个动作的真实价值按照N(0,1)的分布随机生成。对于在t时刻选择的动作
![](https://latex.codecogs.com/png.latex?A_t)
，老虎机返回的奖赏大小按照以
![](https://latex.codecogs.com/png.latex?q_*(A_t))
为均值，方差为1的随机分布产生奖赏反馈给我们。下面我们将尝试
![](https://latex.codecogs.com/png.latex?\epsilon)
设置成不同值时的累计奖赏。当设置成0是表示完全贪心。


