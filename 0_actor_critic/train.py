#################################
## Train
## 1. Run Agent in the Environment, and get some data
## >> 2. Calculate reward for each time steps
## >> 3. Calculate Actor-Critic Model's loss function
## >> 4. Calculate Gradient and update parameters
## 5. Repeat 1-4 until "success"
#################################
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from run import run_episode
from configs import huber_loss, eps

# 2. Calculate reward for each time steps
# (Gt, apply discount factor)
def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
  '''
  Compute expected returns per timestep
  rewards: tf.Tensor, rewards from an episode
  gamma: float, discount factor
  standardize: bool, whether standardize returns or not
  return: tf.Tensor, computed expected returns
  '''
  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of 'rewards' and accumulate reward sums into the 'returns' array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
  
  return returns

# 3. Calculate Loss function of the Actor-Critic Model
'''
L = L_actor + L_critic
1. L_actor
  based on policy gradients with the critic as a state dependent baseline, and per-episode estimates
  L_actor = -sigma(t=1 to T) [log (pi_theta(at|st)) * [G(st, at) - V_theta(st)])]
  T: number of time steps per episode
  st, at: state and selected action at time t
  pi_theta: policy
  V_theta: value function
  G: expected return for given state and action
2. L_critic
  L_critic = L_delta(G, V_theta)
  L_delta: Huber loss (less sensitive to outliers than squared-error loss)
'''
def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
  '''
  Computes combined loss
  action_probs: tf.Tensor
  values: tf.Tensor
  returnsx: tf.Tensor
  return: tf.Tensor
  '''
  advantage = returns - values
  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
  critic_loss = huber_loss(values, returns)
  return actor_loss + critic_loss

# 4. Calculate Gradient and Update Parameters
@tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float, max_steps_per_episode: int) -> tf.Tensor:
  '''
  Runs a model training step
  initial_state: tf.Tensor
  model: tf.keras.Model
  optimizer: tf.keras.optimizers.Optimizer
  gamma: float
  max_steps_per_episode: int
  return: tf.Tensor
  '''
  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)
    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)
    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)
  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)
  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  episode_reward = tf.math.reduce_sum(rewards)
  return episode_reward