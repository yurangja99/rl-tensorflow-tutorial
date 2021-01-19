#################################
## Main
#################################
import tensorflow as tf

from configs import create_env, convert_env_to_tf, create_q_net, create_dqn_agent, create_random_policy, create_replay_buffer
from configs import num_eval_episodes, initial_collect_steps, batch_size
from run import collect_data
from train import compute_avg_return, train
from plot import plot_learning_curve, create_policy_eval_video
from tests import test_env_reset, test_env_time_steps, test_env_time_step_spec, test_policy_action, test_agent_collect_data

def main():
  '''
  main function
  '''
  # two environments. 
  # one for training, and the other for evaluation.
  train_py_env = create_env()
  eval_py_env = create_env()
  train_env = convert_env_to_tf(train_py_env)
  eval_env = convert_env_to_tf(eval_py_env)

  # set train_step_counter
  train_step_counter = tf.Variable(0)

  # define q network for calc action values
  q_net = create_q_net(train_env)

  # define and initialize dqn agent using q network
  agent = create_dqn_agent(train_env, q_net, train_step_counter)
  agent.initialize()

  # define some random policy
  random_policy = create_random_policy(train_env)
  
  # define replay buffer to record data
  replay_buffer = create_replay_buffer(train_env, agent)

  # collect some data to the replay buffer
  # after below command, replay_buffer becomes a list of Trajectories
  collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

  # access to the replay buffer, and get datasets
  # sample_batch_size: just batch size
  # num_steps: number of adjacent rows for each elements in the batch
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)
  
  # train agent
  returns = train(train_env, agent, replay_buffer, iter(dataset), num_eval_episodes)

  # plot learning curve
  plot_learning_curve(returns)

  # make agent's video
  create_policy_eval_video(eval_env, eval_py_env, agent.policy, "trained-agent")

  # make random policy's video
  create_policy_eval_video(eval_env, eval_py_env, random_policy, "random-agent")

if __name__ == "__main__":
  main()