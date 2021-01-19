#################################
## Configs
## 1. Import necessary packages
## 2. Set global configurations
#################################
import gym
import numpy as np
import tensorflow as tf

# Create the environment
env = gym.make("CartPole-v0")

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

# Set Huber Loss (less sensitive for outlier data)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# Set ADAM Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)