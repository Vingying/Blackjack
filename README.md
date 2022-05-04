本实验基于 ```Gym/Blackjack-v1``` 环境进行

## 环境描述

```Blackjack-v1``` 是进行 21 点游戏的一组环境。21 点的游戏规则此处不赘述。本环境中，默认每一种花色对应的手牌有无数张，其中 A 可以既当作 1 点又可以当作 11 点（称为 Usable Jack）。

环境对应的 Reward 如下：

- Lose: -1
- Draw: 0
- Win:
  - +1 without natural blackjack
  - +1.5 with natural blackjack

Agent 的策略有两种：

- 0：停止摸牌
- 1：继续摸一张牌

## 问题求解

由于环境是未知的，所以不能采用 MDP 的方式。此时的环境是 Model-Free 的，需要采用采样的方式来对环境进行探索和利用。常见的方式有 MC 和 TD 等方式。

对采样进行利用时，采用 $\epsilon-\text{Greedy}$ 来进行。即维护一张 Q-Table，在 $[0,1)$ 内生成一个随机实数 $x$，当 $0\le x < \epsilon$ 时，均匀随机选取策略（本环境中即 0 和 1 二选一）；否则，就贪心地选取 Q-Table 中对应值最大的策略。

在本实验中，$\epsilon$ 的取值与 episode 数成线性相关。具体地，$\epsilon(t)=\max\{ 0.15-3\times 10^{-5}\cdot t,\ 0.05\}$，其中 $t$ 表示当前是第 $t$ 个 episode。

$\epsilon-\text{Greedy}$ 代码如下：

```python
    def epsilon_greedy(self, obs, epsilon, episode_id):
        x = np.random.uniform(0, 1)
        eps = max(0.05, epsilon - episode_id * 0.000003)
        print(episode_id, eps)
        if epsilon < 0:
            eps = 0
        if x < 1.0 - eps:
            j = (obs[0], obs[1], int(obs[2]))
            list_Q = self.Q[j]
            action = np.argmax(list_Q)
        else:
            action = np.random.choice(self.n_act)
        return action
```

最后根据训练出来的 Q-Table 进行测试。本实验中，选择 50000 个 episode 进行训练，对 5000 次进行测试，结果如下：

```
==================TEST RESULT==================
win: 1960 lose: 2807 draw: 233
win_rate: 0.392 lose_rate: 0.5614 draw_rate: 0.0466
===============================================
```

详情训练结果见 result.txt

最后对 Q-Table 生成的状态函数 $v_{\pi}(s)=\mathbb{E}[Q_{\pi}(s,a)|\pi,s]$ 绘图，得到的三维图如下所示

![Figure_1.png](https://s2.loli.net/2022/05/04/IB4cgH6ZlVtiCxT.png) 

0~30 轴表示 Agent 手上的点数之和；0~10 轴表示对手展示的牌的点数；z 轴表示该状态对应的价值。