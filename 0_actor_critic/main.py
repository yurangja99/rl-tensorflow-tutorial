#################################
## Main
#################################
import tqdm
import tensorflow as tf

from configs import env, optimizer
from models import ActorCritic
from train import train_step
from save import run_and_save_episode

def main():
  # Set some parameters for ActorCritic class
  num_actions = env.action_space.n # 2
  num_hidden_units = 128

  # Define an instance of ActorCritic class
  model = ActorCritic(num_actions, num_hidden_units)

  # 5. Run the training loop
  max_episodes = 10000
  max_steps_per_episode = 1000

  # Cartpole-v0 is considered solved if average reward is >= 195 over 100 
  # consecutive trials
  reward_threshold = 195
  running_reward = 0

  # Discount factor for future rewards
  gamma = 0.99

  # Start training
  with tqdm.trange(max_episodes) as t:
    for i in t:
      initial_state = tf.constant(env.reset(), dtype=tf.float32)
      episode_reward = int(train_step(
          initial_state, model, optimizer, gamma, max_steps_per_episode))

      running_reward = episode_reward*0.01 + running_reward*.99

      t.set_description(f'Episode {i}')
      t.set_postfix(
          episode_reward=episode_reward, running_reward=running_reward)

      # Show average episode reward every 10 episodes
      if i % 10 == 0:
        pass # print(f'Episode {i}: average reward: {avg_reward}')

      if running_reward > reward_threshold:  
          break

  print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

  # render and save episode
  run_and_save_episode(env, model, max_steps_per_episode)

if __name__ == "__main__":
  main()