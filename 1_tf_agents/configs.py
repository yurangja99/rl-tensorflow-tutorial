#################################
## Configs
## set some hyper parameters, environment, optimizer, and etc
#################################
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

# number of iterations (for training)
num_iterations = 20000

# used to collect data to replay buffer
initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

# set training params (and logging interval)
batch_size = 64
learning_rate = 1e-3
log_interval = 200

# used to calculation of average reward
num_eval_episodes = 10
eval_interval = 1000

# set environment using tf agents
env_name = 'CartPole-v0'
def create_env():
  '''
  create a new environment, and convert it to tf version
  '''
  return suite_gym.load(env_name)
def convert_env_to_tf(env):
  '''
  converting enables Tensor Agent to work with the environment
  '''
  return tf_py_environment.TFPyEnvironment(env)

# set q network for evaluating action values
fc_layer_params = (100,)
def create_q_net(env):
  '''
  create new q network for evaluating action values
  '''
  return q_network.QNetwork(
    env.observation_spec(), 
    env.action_spec(), 
    fc_layer_params=fc_layer_params)

# set optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# set dqn agent
def create_dqn_agent(env, q_net, train_step_counter):
  '''
  using given env, q network, and optimizer, create new dqn agent
  '''
  return dqn_agent.DqnAgent(
    env.time_step_spec(), 
    env.action_spec(), 
    q_network=q_net, 
    optimizer = optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

# set random policy
def create_random_policy(env):
  '''
  return random policy
  '''
  return random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

# set replay buffer
def create_replay_buffer(env, agent):
  return tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=env.batch_size,
    max_length=replay_buffer_max_length)