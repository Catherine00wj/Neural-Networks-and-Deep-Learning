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
BATCH_SIZE = 80 # size of minibatch
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

# TODO: Define Network Graph
# q-train -- network weights-----------------------------------
W1 = tf.Variable(tf.truncated_normal([STATE_DIM, size]))
b1 =  tf.Variable(tf.constant(0.01, shape = [size]))
W2 = tf.Variable(tf.truncated_normal([size, ACTION_DIM]))
b2 = tf.Variable(tf.constant(0.01,shape= [ACTION_DIM]))
# hidden layer----------------------------------
h_layer = tf.nn.relu(tf.matmul(state_in, W1) + b1)
# Q Value layer----------------------------------
q_values =tf.matmul(h_layer, W2) +  tf.cast(b2,float)
q_action = tf.reduce_sum(tf.multiply(q_values,action_in),reduction_indices = 1)
loss = tf.reduce_mean(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# TODO: Network outputs
# q_values =network.q_value
# q_action =create_training_method.q_action

# TODO: Loss/Optimizer Definition
# loss = create_training_method().cost
# optimizer =create_training_method().optimizer

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action



# Main learning loop
for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    # Update epsilon once per episode

    epsilon -= (epsilon  -FINAL_EPSILON)/ EPSILON_DECAY_STEPS
    # Move through env according to e-greedy policy
    for step in range(STEP):
#--------------------------------------------
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.popleft()
        if len(replay_buffer) > BATCH_SIZE:
            time_step += 1
            # Step 1: obtain random minibatch from replay memory
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]
            # Step 2: calculate target
            target_batch = []
            q_values_batch = q_values.eval(feed_dict={state_in: next_state_batch})
            for i in range(0, BATCH_SIZE):
                is_done = minibatch[i][4]
                if is_done:
                    target_batch.append(reward_batch[i])
                else:
                    target_batch.append(reward_batch[i] + GAMMA * np.max(q_values_batch[i]))
            # nextstate_q_values = q_values.eval(feed_dict={
            #     state_in: [next_state]
            # })
            # TODO: Calculate the target q-value.
            # hint1: Bellman
            # hint2: consider if the episode has terminated
            # target = reward + GAMMA * np.max(nextstate_q_values)
            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: target_batch,
                action_in: action_batch,
                state_in: state_batch
            })
        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
