#################################
## Run
#################################
from tf_agents.trajectories import trajectory

def collect_step(env, policy, buffer):
  '''
  in given env, do action according to policy and write it to the buffer
  '''
  # select action and do it.
  time_step = env.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = env.step(action_step.action)

  # get trajectory from s, a, s', r
  traj = trajectory.from_transition(time_step, action_step, next_time_step)
  
  # add it to the buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  '''
  repeat collect_step function for 'steps' steps.
  in given env, do actions according to the policy,
  and record it to the buffer.
  '''
  for _ in range(steps):
    collect_step(env, policy, buffer)