# Project 1: Navigation
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

## Introduction

The aim of this project is to train an agent to navigate in the given environment to collect yellow bananas and avoid purple bananas.

![Trained Agent](/images/TrainedAgent.gif)

*Figure 1: Interaction of a Trained Agent*

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
## Algorithm
### Markov Decision Process and Tabular Method

This project uses a Deep Q Network algorithm which is based on Q-learning implemented using neural networks. Reinforcement learning algorithms such Q learning interacts with the environment and learns from the experience unlike supervised learning. In reinforcement learning methods the algorithm is called an agent. The agent interacts with the environment through actions taken based on observations, which are called states. And the environment provides the agent a feedback through rewards. The agent learns the task by aiming to maximize the rewards. The task can be either episodic for example navigating a robot from point A to point B or the task can be continuous for example balancing an inverted pendulum.


### Deep Q Network Algorithm
### Double Q Network
## Results and Conclusion
