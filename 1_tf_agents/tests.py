#################################
## Tests
## some basic tests for understanding tf-agents
#################################
import numpy as np

def test_env_reset(env):
  '''
  reset given environment
  '''
  env.reset()

def test_env_time_step_spec(env):
  '''
  env.step(action) returns TimeStep tuple
  and, TimeStep tuple contains observation and reward.
  this function display action and those fields.
  '''
  print("Action Spec: ")
  print(env.action_spec())
  print("Observation Spec: ")
  print(env.time_step_spec().observation)
  print("Reward Spec: ")
  print(env.time_step_spec().reward)
  
def test_env_time_steps(env):
  '''
  this function resets given env, and do an action.
  then, two time steps (for reset and given action) returned
  this function displays them.
  '''
  time_step = env.reset()
  print("Time step: ")
  print(time_step)
  action = np.array(1, dtype=np.int32)
  next_time_step = env.step(action)
  print("Next time step: ")
  print(next_time_step)

def test_policy_action(policy, time_step):
  '''
  given policy and time step, the policy returns next action.
  PolicyStep(action, dtype, numpy, state, info)
  '''
  print(policy.action(time_step))

def test_agent_collect_data(agent):
  '''
  given agent, print collect_data_spec
  '''
  print(agent.collect_data_spec)
  print(agent.collect_data_spec._fields)