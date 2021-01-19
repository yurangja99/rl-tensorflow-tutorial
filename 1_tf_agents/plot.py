#################################
## Plot
## plot learning curve
#################################
import matplotlib.pyplot as plt
import imageio

from configs import num_iterations, eval_interval

def plot_learning_curve(returns):
  '''
  plot learning curve using training history.
  '''
  iterations = range(0, num_iterations + 1, eval_interval)
  plt.plot(iterations, returns)
  plt.ylabel('Average Return')
  plt.xlabel('Iterations')
  plt.ylim(top=250)

def create_policy_eval_video(env, py_env, policy, filename, num_episodes=5, fps=30):
  '''
  create video of the agent
  '''
  filename = filename + ".mp4"

  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = env.reset()
      video.append_data(py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = env.step(action_step.action)
        video.append_data(py_env.render())