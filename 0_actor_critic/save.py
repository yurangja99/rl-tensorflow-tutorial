#################################
## Save
## Given env, model, and max steps, run simulation and save it.
#################################
import gym
import numpy as np
import tensorflow as tf
from PIL import Image

# Render an episode and save as a GIF file
def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int): 
  screen = env.render(mode='rgb_array')
  im = Image.fromarray(screen)

  images = [im]

  state = tf.constant(env.reset(), dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, 0)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, _, done, _ = env.step(action)
    state = tf.constant(state, dtype=tf.float32)

    # Render screen every 10 steps
    if i % 10 == 0:
      screen = env.render(mode='rgb_array')
      images.append(Image.fromarray(screen))

    if done:
      break

  return images

def run_and_save_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
  # Save GIF image
  images = render_episode(env, model, max_steps)
  image_file = 'cartpole-v0.gif'
  # loop=0: loop forever, duration=1: play each frame for 1ms
  images[0].save(
      image_file, save_all=True, append_images=images[1:], loop=0, duration=1)

  import tensorflow_docs.vis.embed as embed
  embed.embed_file(image_file)