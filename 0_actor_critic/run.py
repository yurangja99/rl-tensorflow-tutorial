#################################
## Train
## >> 1. Run Agent in the Environment, and get some data
## 2. Calculate reward for each time steps
## 3. Calculate Actor-Critic Model's loss function
## 4. Calculate Gradient and update parameters
## 5. Repeat 1-4 until "success"
#################################
import numpy as np
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

from configs import env

# 1. Get Data
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''
  Get action, and return 
  action: np.ndarray, action
  return: Tuple(np.ndarray), state, reward, and done flag
  '''
  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  '''
  Wrapper function of env_step for Tensorflow Tensor
  action: tf.Tensor, action
  return: List(tf.Tensor), returned value of env_step(state, reward, and done flag)
  '''
  return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])

def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int) -> List[tf.Tensor]:
  '''
  Runs a single episode (1 episode) to collect some data
  initial_state: tf.Tensor, initial state (S0)
  model: tf.keras.Model, our agent
  max_steps: int, max time step before terminating the episode
  return: List(tf.Tensor), data from the episode
  '''
  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)
    # Store critic values
    values = values.write(t, tf.squeeze(value))
    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])
    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)
    # Store reward
    rewards = rewards.write(t, reward)
    # Break if done flag is enabled
    if tf.cast(done, tf.bool):
      break
  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()
  return action_probs, values, rewards