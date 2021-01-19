#################################
## Models
## Build Actor-Critic Model
#################################
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple

class ActorCritic(tf.keras.Model):
  '''
  Combined Actor-Critic Network
  In this network, Actor calculates probabilities that this policy selects each actions.
  Also, Critic calculates value function for given state.
  '''
  def __init__(self, num_actions: int, num_hidden_units: int):
    '''
    Initialize This Actor-Critic Method
    num_actions: int, number of actions
    num_hidden_units: int, number of hidden nodes in a hidden layer
    return: ()
    '''
    super().__init__()

    # Set Hidden Layer, Actor, and Critic
    # In this setting, Actor and Critic share a hidden layer called "common"
    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)
  
  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    '''
    From input (tf.Tensor), calculate policy and value function
    inputs: tf.Tensor, approximates a state (position, velocity, angle, and angle velocity)
    return: Tuple(tf.Tensor), calculated probabilities for each actions (Actor), and value function for given state (Critic)
    '''
    x = self.common(inputs)
    return self.actor(x), self.critic(x)