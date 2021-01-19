#################################
## Train
## calculate some values and train agent
#################################
from tf_agents.utils import common

from configs import num_iterations, collect_steps_per_iteration, log_interval, eval_interval
from run import collect_data

def compute_avg_return(env, policy, num_episodes=10):
  '''
  through simulating episodes, calculates average reward of the policy
  '''
  total_return = 0.0
  for _ in range(num_episodes):
    # reset env and set episode return to zero
    time_step = env.reset()
    episode_return = 0.0
    
    while not time_step.is_last():
      # select action, do it, and get return
      action_step = policy.action(time_step)
      time_step = env.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return
  
  # calculate average return and return
  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

def train(env, agent, replay_buffer, dataset_iterator, num_eval_episodes):
  '''
  train agent
  '''
  # optimize by wrapping some of the code in a graph using TF function.
  # (이게 뭔소리인가?)
  agent.train = common.function(agent.train)

  # reset the train step
  agent.train_step_counter.assign(0)

  # evaluate the agent's policy once before training.
  avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
  returns = [avg_return]

  # start training for some steps
  for _ in range(num_iterations):
    # collect some data and save to the replay buffer
    collect_data(env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # sample a batch of data from the buffer and update the agent's network
    experience, unused_info = next(dataset_iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print("step = {0}: loss = {1}".format(step, train_loss))
    if step % eval_interval == 0:
      avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
      print("step = {0}: Average Return = {1}".format(step, avg_return))
      returns.append(avg_return)
  return returns
