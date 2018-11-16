import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9  # discount factor
INITIAL_EPSILON = 0.6# starting value of epsilon
FINAL_EPSILON =0.01# final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
replay_buffer = deque()
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 90 # size of minibatch
time_step =0
size = 100

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

