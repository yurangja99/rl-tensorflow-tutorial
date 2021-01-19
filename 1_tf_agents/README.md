# Tensorflow Agents (DQN)
- link: https://www.tensorflow.org/agents/overview?hl=ko

## Cartpole Environment
- same as the 0_actor_critic
- observation: 4d float array. cart's pos and vel, pole's pos and vel
- reward: fixed scalar value
- action: scalar value. 0 or 1

## Differences from 0_actor_critic

### Environment
- In 0_actor_critic, we used env from gym. In this project, we will use tf_agents.environments.suite_gym
- environment.step(): action -> TimeStep [which contains next observation (or state), and reward]
- time_step_spec(): TimeStep tuple -> observation, and reward

### DQN Algorithm
- DQN Algorithm (uses q network for evaluating action value)
- Replay Buffer

## Files
- main.py: main process
- configs.py: some global configurations, and hyper parameters
- run.py: runs episodes to collect data
- train.py: calculates some values for training, and train agent
- plot.py: plot learning curve, etc
- tests.py: some basic test methods