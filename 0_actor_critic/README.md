# Actor-Critic Method using Tensorflow
- link: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic?hl=ko

## CartPole Play with TF Actor-Critic Method
- Tensorflow
- Actor-Critic Method
- Open AI Gym CartPole-V0 Environment
- Policy Gradient

## CartPole-v0
- horizontal track with no friction
- agent forces -1 or +1
- reward = +1 for standing
- terminates when angle >= 15 or abs(x) >= 2.4
- assumed as "solved" when average total reward for the episode reaches 195 over 100 consecutive trials

## Files
- main.py: main process
- configs.py: configurations such as random seeds, environment, optimizer, etc
- models.py: Actor-Critic Model
- run.py: run episode and generates data
- train.py: train model using generated data
- save.py: run episode and save it as gif file