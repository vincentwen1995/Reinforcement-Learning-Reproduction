# Reinforcement Learning Reproducibility Lab

## Configuration
Install the conda environment by running `conda env create -f environment.yml`

## Run
Run the preferred module by `python -m <module_name>` with speicified arguments.

## The topic we choose
N-step bootstrapping in actor-critic methods.

## Motivation and research question
In this project, we study n-step bootstrapping in actor critic methods, more specific, we study advantage actor critic (A2C). 

### N-step bootstrapping
N-step bootstrapping, or TD(N) is a very important technique in Reinforcement Learning that performs update based on intermediate number of rewards. In this view, N-step bootstrapping unifies and generalizes the Monte Carlo (MC) methods and Temporal Difference (TD) methods. From one extreme, when $N=1$, it is equivalent to TD(1), from another extreme, when $N=\infty$, i.e., taking as many steps as possible until the end of the episode, it becomes MC. As a result, N-step bootstrapping also combines the advantages of Monte Carlo	and	1-step TD. Compared to 1-step TD, n-step bootstrapping will	converge faster because it bootstraps with more real information and it is freed from the “tyranny of the time step”. Compared to MC, the updates do not have to wait until the end of the episode and it is also more efficient and less variants. In general, when facing different problems / situations, with a suitable N, we could often achieve faster and more stable learning.


<div align="center"><img src="https://img-blog.csdnimg.cn/20191014053755194.png" width="300" height="300" alt="N-steps" align=center></div>

### Advantage Actor Critic (A2C)
Actor-Critic algorithms are a power families of learning algorithms within the policy-based framework in Reinforcement Learning. It composes of actor, the policy that makes decision and critic, the value function that evaluates if it is a good decision. With the assistant from critic, the actor can usually achieves better performance, such as by reducing gradient variance in vanilla policy gradients. From the GAE paper , John Schulman has unified the framework for advantage estimation, between all the GAE variants, we picked A2C considering the amazing performance of A3C and it is a simplified version of A3C with equivalent performance.

## N-step bootstrapping for A2C
A2C is an online algorithm that uses roll-outs of size k + 1 of the current policy to perform a policy improvement step. In order to train the policy-head, an approximation of the policy-gradient is computed for each state of the roll-out $\left(x_{t+i}, a_{t+i} \sim \pi\left(\cdot | x_{t+i} ; \theta_{\pi}\right), r_{t+i}\right)_{i=0}^{k}$:
$$
    \nabla_{\theta_{\pi}} \log \left(\pi\left(a_{t+i} | x_{t+i} ; \theta_{\pi}\right)\right)\left[\hat{Q}_{i}-V\left(x_{t+i} ; \theta_{V}\right)\right]
$$

## Experiments
For this project, our main goal is to compare the performance of the n-step bootstrapping variation of the $\textcolor{red}{METHODS}$ with its Monte-Carlo and 1-step variations. Therefore, we do not seek to deliver an ultimate agent that can solve some complicated fancy games.

The experiments are designed with classical control problems, i.e. InvertedPendulum, CartPole, Acrobot, MountainCar etc. For the sake of convenience in terms of implementation, we use the off-the-shelf environments provided by $\href{https://gym.openai.com/}{ OpenAI's Gym library}$. Under the category $\textit{Classical Control}$, we picked out two discrete environments: $\textbf{CartPole-v0}$ and $\textbf{Acrobot-v1}$.
