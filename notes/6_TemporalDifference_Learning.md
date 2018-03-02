# 时序差分学习 (Temporal-Difference Learning)

---

如果说强化学习中存在一个最核心的概念，那一定是时许差分学习，又称为TD learning。

TD learning是MC思想和DP思想的结合。它可以像MC方法似的直接从raw experiment中学习，也可以像DP方法一样基于之前的估计进行新的估计，
而不用像MC一样必须等到一个episode结束才能更新。

一般情况下，我们首先关心的是policy evaluation，也就是prediction problem，就是给定policy估计value function。
对于control problem，目的是找到一个optimal policy，此时DP、TD、MC方法都是GPI问题的变体。
这三种方法的主要不同之处就是prediction的方法不同。

# 1. TD预测 (TD Prediction)
一个简单的every-visit MC方法用于nonstationary environment的公式为：

<center>

![](../images/6_TemporalDifference_Learning/mc_nonstationary.PNG)

</center>

这种方法被称为常数步长的MC方法(
![](https://latex.codecogs.com/png.latex?constant-\alpha)
MC)。

从上式可以看出，MC方法必须等到episode技术才可以增量更新V，因为需要等到
![](https://latex.codecogs.com/png.latex?G_t)
已知才可以。而TD方法只需要等到下一个time step就可以。

在t+1时刻，我们立刻就得到了一个target，就是使用observed reward
![](https://latex.codecogs.com/png.latex?R_{t+1})
和估计值
![](https://latex.codecogs.com/png.latex?V_{t+1})
来更新V。最简单的TD更新形式为：
<center>

![](../images/6_TemporalDifference_Learning/TD_update.PNG)

</center>

MC的target是
![](https://latex.codecogs.com/png.latex?G_t)
，就是MC是朝着累积奖赏的方向更新的。而TD是朝着立刻得到的target，就是
![](https://latex.codecogs.com/png.latex?R_{t+1}+\gamma V(S_{t+1}))
，这种TD方法称为**TD(0)**，或者一步TD(**one-step TD**)，
因为这个方法是
![](https://latex.codecogs.com/png.latex?TD(\lambda))
和n-step TD方法的特例。

下面给出TD(0)的伪代码形式如下：
<center>

![](../images/6_TemporalDifference_Learning/TD_zero.PNG)

</center>

我们可以根据第三章知识得到：
<center>

![](../images/6_TemporalDifference_Learning/dp_derivation.PNG)

</center>

这里，MC方法是把第一个等号作为了target，而DP则是把第三个等号作为target。

MC方法的target是一个估计是因为式子中的期望未知，需要估计期望；
DP方法的target是一个估计并不是因为期望，因为期望是可以精确计算的，因为给出了环境的状态转移。
它之所以是一个估计是因为
![](https://latex.codecogs.com/png.latex?v_\pi(S_{t+1}))
是未知的，这里是使用当前的估计：
![](https://latex.codecogs.com/png.latex?v(S_{t+1}))来估计的。

TD target是一个估计就有上面两点原因:采样期望值，同时使用当前估计V，而不是true 
![](https://latex.codecogs.com/png.latex?v_\pi)
。因此，TD方法结合了MC的sampling和DP的bootstapping。

下面给出TD(0)的backup diagram：

<center>

![](../images/6_TemporalDifference_Learning/backup.PNG)

</center>

TD和MC的更新方式称为采样更新(sample updates)，DP的更新方式称为期望更新(expected updates)。
sample updates意思就是需要涉及向前走去采样记下来的状态(successor state)。

在上面的伪代码中，可以看到括号内的差值(difference)，是状态
![](https://latex.codecogs.com/png.latex?S_t)
的estimated value，和在向前看了一步之后的better estimate
![](https://latex.codecogs.com/png.latex?R_{t+1}+\gamma V(S_{t+1}))
，这个差值称为TD误差(**TD error**)。这个TD error会以各种形式在强化学习中出现：

<center>

![](../images/6_TemporalDifference_Learning/TD_error.PNG)

</center>

注意，在每个时刻的TD error是实时得到的(made at that time)，因为TD error依赖于next state和next reward。
如果不look ahead，我们是不能得到这两个的。这就是说
![](https://latex.codecogs.com/png.latex?\delta_t)
是状态
![](https://latex.codecogs.com/png.latex?S_t)
的值估计
![](https://latex.codecogs.com/png.latex?V(S_t))
在t+1时刻得到的误差(error)。

注意，如果在一个episode期间，V数据保持不变的话，那么MC error可以写成TD error的和，如下所示：
<center>

![](../images/6_TemporalDifference_Learning/MC_error.PNG)

</center>

这个推导不太准确，原因是TD(0)的在episode期间会更新的，只能说如果step size很小的话，V可以保持大概不变(hold approximately)。

上面这个推导的泛化在TD learning中起到了重要作用。

TD learning就是要在采样的过程中，对于当前不太准的估计进行及时更新。下面举一个形象的例子来说明这个过程。

这个例子就是开车回家的例子(Driving Home)。
每天下班后你都会开车从公司回家，因此你想要预测什么时候能到家。当你离开办公室后，你考虑了今天是周几、天气如何、是不是晚高峰以及其他相关的因素,
最后你估计你需要30分钟的时间到家。这时你看了一眼表，正好6点整。

然后你从公司走到停车场找到你的车，这时看了下表，6:05，但是这时开始下雨了。你知道下雨的话车走的就慢了，还可能堵车，
因此你重新估计了时间，你觉得从现在开始还需要35分钟才能到家，或者说总共需要40分钟（加上刚才找车的5分钟)。

15分钟后你走完了高速，一路还比较顺畅。这时候你走上辅路准备回家，你觉得快到了，因此你现在觉得总过用35分钟就能到家了。

但是这时你发现你正跟着一个大卡车，这个大卡车把路占满了，因为路比较窄，还不能超车。因此你只能跟着这个大卡车。
最后6:40终于换到另外一条路上了。3分钟后你到家了。

上面的过程可以用下面的表格表示
<center>

![](../images/6_TemporalDifference_Learning/driving_home_table.PNG)

</center>

我们设置disouting
![](https://latex.codecogs.com/png.latex?\gamma=1)
。因此每个状态的return就是从这个state开始的actual time。每个状态的value就是期望要走的时间。
表格的第二列数字给出了遇到每个状态的current estimated value。

我们把最后一列数字，就是你预测的总共花费的时间画成折线图。看下MC方法是如何操作的。
<center>

![](../images/6_TemporalDifference_Learning/MC_plot.PNG)

</center>

上图折线上的点表示每个状态下我们对于总花费时间的估计。虚线表示实际用时，也就是MC方法的target，最后的return。
在完成这个episode之后，就是更新每个state的value的时候了。比如在leaving office状态下，我们估计的耗时是30分钟，
而实际是43分钟，这说明我们的估计不合理啊，因此根据上面的constant alpha MC方法，假设
![](https://latex.codecogs.com/png.latex?\alpha=1)
，则更新当前的估计为V(leaving office)=43。

所以图中的箭头就是prediction的改变值。

比如在离开高速公路时，我们估计的总耗时是35分钟，但实际却用了43分钟，中间差了8分钟。如果step size是0.5的话，那最后我们的估计就要增大4分钟。
这个其实是一个比较大的更新，可能跟在大车后面只是一个偶然事件。

使用MC方法，所有的改变必须是离线的(off-line)的，就是必须到家之后才能更新，因为只有到家了才知道actual return。

但是等到最后的结果出来再开始学习真的有必要么？比如又有一天你从公司出来，估计30分钟能到家，但是你在高速公路上堵车了，
现在你估计可能得用50分钟才能到家了。在你堵车的时候你没事就会想，是不是之前我估计的30分钟太乐观了。。。

所以我们必须要等到到家了才能更新么？MC方法下你必须这么做，但是TD learning方法下你不用等到回家再更新。

按照TD方法，你可以立刻把初始状态估计的30分钟向着50分钟移动一点。

下面的图给出了之前开车的例子使用TD方法的TD规则的prediction的更新过程：
<center>

![](../images/6_TemporalDifference_Learning/TD_plot.PNG)

</center>

每个error按照一定的比例去更新prediction，这就是在prediction中的时序差分(Temporal Difference)。

# 2. TD预测方法的优势(Advantages of TD Prediction Methods)

TD方法利用猜测进行猜测(learn a guess from a guess)，也就是bootstrap。那么TD方法比起MC和DP方法有什么优势呢？

首先，TD比DP方法的一个优势就是TD不要求environment model。

另一个TD方法比DP方法的一个优势是TD方法容易on-line的实现，不用等到一个episode结束再更新。这个有时是一个十分重要的优势，
因为有些application的episode很长，如果等到这个episode结束再更新就太慢了。还有些应用是continuing task，要是MC方法就得使用discount episodes等。
学习较慢。但是TD方法不管你episode是不是无穷，长度是多少，我只需要看你下一步转移到哪就行了。

虽然我们使用one guess来估计value确实方便，但是这种方法能保证收敛到正确的结果吗？答案是yes。
对于任何固定的policy
![](https://latex.codecogs.com/png.latex?\pi)
，TD(0)已经被证明可以收敛到
![](https://latex.codecogs.com/png.latex?v_\pi)

如果TD和MC方法都能收敛到correct predictions，那么我们应该首先选择哪种方法呢？
目前没有人能从数学上证明一个方法比另一个收敛的快。但是实际上，TD方法一般在stochastic tasks上比constant-alpha MC收敛的快。

下面举例说明。下图表示了一个马尔科夫奖赏过程(Markov reward process, MRP)：

<center>

![](../images/6_TemporalDifference_Learning/markov_reward_process.PNG)

</center>

MRP其实就是没有action的MDP。下面我们使用MRP关注prediction问题。

在MRP中，所有的起始点都是中间state C。下面就是每步等概率的选择向左或者向右走。最后的终态是方块，走到左边的方块则return为0，
走到右边的方块则return为1。并且这个task是undiscounted的。每个状态的true value就是从当前状态走到右边状态的概率。
因此我们很容易知道从状态A到E，这5个state的true value分别是
![](https://latex.codecogs.com/png.latex?\frac{1}{6},\frac{2}{6},\frac{3}{6},\frac{4}{6},\frac{5}{6})。

下面给出使用constant step-size为0.1时候的TD(0)方法。在不同的episode更新之后的结果如下图所示：
<center>

![](../images/6_TemporalDifference_Learning/TD_zero_episodes.PNG)

</center>

可以看出在100个episode更新之后估计的值越来越接近true value。

下面的图展示了不同alpha下MC方法和TD方法的估计值与true value之间的均方根误差(root mean-squared error, RMSE)。
<center>

![](../images/6_TemporalDifference_Learning/compare_episodes.PNG)

</center>

上面的估计初始值都是0.5，从上图可以看出TD方法比MC方法效果要好。并且还有一个现象就是越往后的episode，波动越明显。