# Monte Carlo Methods

---

之前的DP方法是直接在完全知道环境的情况下，通过迭代，计算在每个状态的值，然后得到最优策略。

# 1. 蒙特卡洛预测(Monte Carlo Prediction)

这一章，我们将首次尝试通过经验(experience)估计value。直观的方法就是将从当前state之后的所有访问(visit)得到的returns进行简单平均。
当returns被观测的次数越来越多时，平均的结果将会收敛到期望值。这类方法称为蒙特卡洛方法(Monte Carlo method)。

在一个episode中，每次访问状态s，叫做对状态s的一次访问(visit)。在同一个episode中，状态s可以被访问多次。

**first-visit MC**方法是将第一次对s进行visit之后的returns计算平均。而**every-visit MC**方法是将对状态s的所有visit之后的returns
计算平均。这两种方法相似但是理论特性不同。我们这一章主要关注的是first-visit MC方法。而every-visit MC方法更适用于function approximation
和eligibility traces。

下面给出first-visit MC方法的伪代码形式

<center>

![](../images/5_monte_carlo_methods/first_visit_mc.PNG)

</center>

上述伪代码中，我们在初始化之后，在每次循环中，要先使用policy生成一个episode，然后对于episode中出现的每个状态s，把状态s第一次出现
的累积回报加入到状态s的回报列表中，然后计算平均值作为对状态s的value的估计。

Every-visit MC方法没有这么直观，但是有论文指出every-visit可以二阶收敛(converge quadratically)到
![](https://latex.codecogs.com/png.latex?v_\pi(s))

MC方法能否画出backup diagram？实际是可以的。如下图所示。

<center>

![](../images/5_monte_carlo_methods/mc_backup.PNG)

</center>

在第3章时候给出了DP的diagram，如下图所示

<center>

![](../images/5_monte_carlo_methods/dp_backup.PNG)

</center>

DP的diagram展示了所有的转移可能，而MC的diagram只展示出被采样到的那种可能；
DP的diagram只包括一步转移，而MC的diagram一直持续到episode结束。

需要注意的是，MC方法对于每个状态的估计是独立的，即对于当前状态的估计不依赖于其他的任何状态的估计，因此MC方法不是bootstrap的。
而DP方法则是bootstrap的方法。

因此估计一个状态的value的计算代价是独立于state的个数的。比如我们想估计状态s的value，我们只需要从状态s开始生成多个episode，直接
计算returns的平均就可以，不需要知道中间经过的每个状态的value的估计。因此计算效率高。

# 2. 状态动作值的蒙特卡洛估计(Monte Carlo Estimation of Action Values)

如果模型未知，那么估计动作值函数(Action value)--就是状态动作对(state-action pairs)--要比估计state value有用的多。

当有模型的时候，只有state value就足够确定policy了，我们可以往前看一步(look ahead one step)，找到哪个动作可以得到最好的reward和
next state，这也是我们在DP方法中所使用的方法。

但是如果模型未知，那么只使用state value就不够了。我们必须显式地给出每个action value的估计，以便于我们能根据action value确定policy。

因此，MC方法的一个主要目标就是估计
![](https://latex.codecogs.com/png.latex?q_*)
，为了实现这一目标，我们首先要对action value进行policy evaluation。

对于action value的policy evaluation问题实际就是估计
![](https://latex.codecogs.com/png.latex?q_\pi(s,a))
。无论对于every-visit MC 方法还是first-visit MC方法，思路都是和对于state value的估计差不多的。
对于一个episode，找到每个state-action pair s,a的visit，并计算多个episode的平均returns。

看起来很完美，但是唯一的问题就是可能有许多的state-action pairs从来没有被visit过。尤其是如果policy是确定性策略(deterministic policy),
那么按照这个policy，对于每个状态只能出现一种动作。这个问题很严重，因为我们估计action value的目的就是帮我们从action中选择一个比较好的。
因此，对于每个状态，我们需要估计所有的action的value。

上面的问题一般称为保持探索(maintaining exploration)。一种可行方法是我们对于一个episode，固定从一个state-action pair开始，
每个对都有非0的概率作为episode的开始。这样在无限个episode中，一定会对每个对都得到action value的估计。这种方法称为
探索性开始(exploring start)。

exploring start有时候是有用的，但是不能通用，尤其是在与环境的交互中直接学习时。因此一般的方法就是需要保证所有的state-action pair都是
可以遇到的，即只考虑在每个state下所有action都有非0的概率被选择的stochastic的情况。后面两章考虑的时这种stochastic方法的变体。
这里我们先使用exploring start。

# 3. 蒙特卡洛控制(Monte Carlo Control)

首先考虑经典policy iteration的MC版本。这种方法中，我们交替使用多步policy evaluation和policy improvement。从任意policy开始，
最终得到optimal policy和optimal action-value function。过程示意如下：

<center>

![](../images/5_monte_carlo_methods/mc_process.PNG)

</center>

其中E表示完成的policy evaluation，I表示完整的policy improvement。policy evaluation前面已经提到了方法，policy improvement则
是使用policy greedy方法。具体公式为：

<center>

![](../images/5_monte_carlo_methods/policy_improvement.PNG)

</center>

下面我们做出两个实际中不太可能成立的假设：
+ 每个episode都使用exploring starts
+ policy evaluation可以使用无限的episodes

在实际使用的过程中，我们必须要想办法去掉这两个假设。下一节将介绍如何去掉第一个假设。

对于第二个假设，这个比较容易移除。有两种考虑思路。

第一种是认为policy evaluation是渐进地近似于真正的value function。在DP和MC中都有这个问题。无限的episode可以保证更小的估计误差，
但是在我们能够接受的误差范围内，使用有限的episode个数是可以的。

第二种是放弃在policy improvement之前的完整的policy evaluation过程。这种观点的极端形式就是value iteration，在policy improvement之前
只进行一次policy evaluation。在MC中的表现就是放弃无限的episode假设。

MC policy evaluation可以基于episode进行交替，叫做**Monte Carlo ES**方法，就是包含exploring starts的MC方法。过程如下：

<center>

![](../images/5_monte_carlo_methods/mc_es.PNG)

</center>

# 4. 不带ES的MC控制(Monte Carlo Control without Exploring Starts)

如何避免exploring start这个不合理的假设？
唯一的方法就是使每个action都能被选中，有两种方法可以做到这一点：
+ 在线方法(on-policy method)
+ 离线方法(off-policy method)

On-policy方法是指evaluate和improve的policy与生成episode的数据一样；Off-policy方法就是evaluate和improve的policy与生成数据
的方法不同。MC ES方法是on-policy的方法，因为评估和提升的policy与生成数据的policy是一样的。

这一节先介绍on-policy的MC控制方法用来消除exploring starts假设。Off-policy的方法下一节再讲。

在on-policy控制方法中，policy一般是软的(soft)，即对于任意state s，任意action a，都有
![](https://latex.codecogs.com/png.latex?\pi(a|s)>0)
。并且一般情况下非常接近于确定性最优策略(deterministic optimal policy)，就是在某个state下，只有一个action的概率特别大，比如0.99，剩下所有的
action概率特别小。这里我们使用的on-policy策略是
![](https://latex.codecogs.com/png.latex?\epsilon-greedy)
策略。即大多数时候选择当前最大action value的action，但是有
![](https://latex.codecogs.com/png.latex?\epsilon)
的概率随机选择其他action。因此所有的nongreedy actions都给一个小的选择概率
![](https://latex.codecogs.com/png.latex?\frac{\epsilon}{|A(s)|})
，大部分情况下选择greedy action，概率是
![](https://latex.codecogs.com/png.latex?1-\epsilon+\frac{\epsilon}{|A(s)|})
。而
![](https://latex.codecogs.com/png.latex?\epsilon-greedy)
方法是
![](https://latex.codecogs.com/png.latex?\epsilon-soft)
的一个例子，
![](https://latex.codecogs.com/png.latex?\epsilon-soft)
表示所有state的所有action的被选择概率满足
![](https://latex.codecogs.com/png.latex?\pi(a|s)>=\frac{\epsilon}{|A(s)|})
。因此可知，在所有的
![](https://latex.codecogs.com/png.latex?\epsilon-soft)
中，
![](https://latex.codecogs.com/png.latex?\epsilon-greedy)
是最接近于greedy的策略。

完整的on-policy first-visit MC control方法如下所示：
<center>

![](../images/5_monte_carlo_methods/on_policy.PNG)

</center>

可以证明最终使用
![](https://latex.codecogs.com/png.latex?\epsilon-soft)
的方法可以收敛到optimal policy，具体证明就不再写了。

到此我们可以只能在所有的
![](https://latex.codecogs.com/png.latex?\epsilon-soft)
策略中找到best policy。但是另一方面，我们也可以消除exploring starts假设了。

# 5. 通过重要性采样进行离线预测(Off-policy Prediction via Importance Sampling)

所有的控制方法都面临一个窘境：我们想要找的是在最优动作下的action value，但是却需要探索所有的non-optimal策略去找optimal action。
那么我们怎么才能按照exploratory policy生成数据却能够学习optimal policy呢？

On-policy方法做了一个妥协--我们也不学习optimal policy了，而是学习一个near-optimal policy，这个policy也在explore。

另外一种思路是我们使用两个policy，一个policy用于学习action value，另一个policy包含exploratory，用来生成行为。
用于学习的policy称为目标策略(target policy)，用于生成行为的policy称为行为策略(behavior policy)。
在这种情况下，我们从那些不是来自target policy的data中学习，这个过程称为**off-policy learning**。

On-policy方法和Off-policy方法各有优劣。On-policy方法通常更简单。Off-policy方法需要额外的概念和表示，并且由于数据来自不同的policy，
因此off-policy方法有更大的variance，并且收敛速度较慢。但是off-policy方法更加强大更加通用。
可以认为on-policy方法是off-policy方法的target policy和behavior policy相同的特殊情况。

在实际应用中，off-policy方法可以有其他变体，比如数据可以来自一个non-learning的控制器，或者来自human expert。
所以Off-policy模型是一种更加general的模型。

这一节我们通过考虑prediction problem来学习off-policy方法。其中target policy和behavior policy是固定的。假设我们希望估计的是
![](https://latex.codecogs.com/png.latex?v_\pi)
或者
![](https://latex.codecogs.com/png.latex?q_\pi)
，但是所有的episodes来自另一个policy b，其中
![](https://latex.codecogs.com/png.latex?b\neq\pi)
。在这个例子中，
![](https://latex.codecogs.com/png.latex?\pi)
是target policy，b是behavior policy。两个policy都是固定已知的。

为了使用从policy b生成的episodes来估计policy 
![](https://latex.codecogs.com/png.latex?\pi)
的value，我们需要满足在target policy中出现的动作，在behavior policy下也会偶然出现。即要求如果
![](https://latex.codecogs.com/png.latex?\pi(a|s)>0)
，则
![](https://latex.codecogs.com/png.latex?b(a|s)>0)
，这叫做覆盖(coverage)假设。
因此behavior policy一定是stochastic的，target policy可能是deterministic的。
这里我们首先考虑的是prediction问题，就是给定的policy，预测action value。因此target policy 
![](https://latex.codecogs.com/png.latex?\pi)
是固定不变的并且已知。

几乎所有的off-policy方法都利用重要性采样(importance sampling)。importance sampling是一个通用的方法，用来通过一个分布的采样估计
另一个分布的期望值的技术。

假设从state 
![](https://latex.codecogs.com/png.latex?S_t)开始，后面按照任一policy
![](https://latex.codecogs.com/png.latex?\pi)
的轨迹是
![](https://latex.codecogs.com/png.latex?A_t,S_{t+1},A_{t+1},...,S_T)
，可以得到：

<center>

![](../images/5_monte_carlo_methods/trajectory.PNG)

</center>

其中p是state-transition probability。

target policy和behavior policy得到的trajectory的相对概率称为重要性采样比率(importance-sampling ratio)，为：

<center>

![](../images/5_monte_carlo_methods/importance_sampling_ratio.PNG)

</center>

可以看出，尽管trajectory的概率依赖于MDP状态转移概率，而这个状态转移概率通常是unknown的，但是分子分母的转移概率抵消了。
因此importance sampling ratio最终只依赖这两个policy和sequence，不依赖MDP。

现在假设我们有一批(a batch)使用策略b生成的的episodes，用来估计
![](https://latex.codecogs.com/png.latex?v_\pi(s))
。这里我们把这批episodes连成一个episode。比如第一个episode的终态的step是100，那下一个episode的开始就是t=101。
这样可以让我们通过time step就知道这是哪个episode的哪个step。

我们定义所有state s被visit的step为集合
![](https://latex.codecogs.com/png.latex?\tau(s))
，这是一个every-visit方法。对于first-visit方法是类似的，就是只包括每个episode第一次visit的time step。同时，令
![](https://latex.codecogs.com/png.latex?T(t))
表示从time t开始第一次termination的time，
![](https://latex.codecogs.com/png.latex?G_t)
表示在t之后直到T(t)的return。对所有
![](https://latex.codecogs.com/png.latex?t\in\tau(s))
，
![](https://latex.codecogs.com/png.latex?G_t)
表示所有出现s的return。
![](../images/5_monte_carlo_methods/ratio.PNG)
表示对应的importance-sampling ratios。为了估计
![](https://latex.codecogs.com/png.latex?v_\pi(s))
，我们对结果做简单平均，称为ordinary importance sampling，公式为：
<center>

![](../images/5_monte_carlo_methods/ordinary_importance_sampling.PNG)

</center>

另外一种方案称为weighted importance sampling，公式为：
<center>

![](../images/5_monte_carlo_methods/weighted_importance_sampling.PNG)

</center>

当分母为0时，定义结果为0。

为了便于理解上面两种importance sampling方法，我们假设只观察到一个return就开始估计。
意思就是没有求和符号了。

这时候，对于weighted average，ratio在分子分母上的抵消了，因此最后的估计就等于观察到的return，与ratio无关。
假设我们只观察到了一次return，那这是一个合理的估计，但是这个是
![](https://latex.codecogs.com/png.latex?v_b(s))
的估计，而不是
![](https://latex.codecogs.com/png.latex?v_\pi(s))
的，因此在统计上是有偏估计(biased)。

但是对于simple average的情况，
![](https://latex.codecogs.com/png.latex?v_\pi(s))
估计是无偏的(unbiased)，但是比较极端(extreme)。比如假设ratio是10，意思就是这个trajectory如果按照target policy被观察到的概率
是在behavior policy下被观察到的概率的10倍。这时ordinary importance-sampling估计的结果是obseved return的10倍。
这就是说尽管这个episode的trajectory在target policy中是一种具有代表性的trajectory，但是我们的估计却远高于实际的observed return。

因此，这两种importance sampling方法的不同之处就是biases和variances。
ordinary importance-sampling估计是unbiased，weighted importance-sampling是biased。
另外，前者的variance是无界的(unbounded)，后者在任何single return上的最大权重是1。

事实上，假设return是有界的(bounded)，即使ratio的方差是无穷的，weighted importance-sampling估计的方差也能收敛到0。
在实际中，weighted estimator通常有很低的variance，一般就用这种。那为什么还要介绍ordinary importance-sampling呢？
是因为ordinary的更容易推广到使用function approximation的approximate method上。

完整的every-visit MC 使用weighted importance sampling进行off-policy evaluation的算法见下一小节，因为设计增量计算。
