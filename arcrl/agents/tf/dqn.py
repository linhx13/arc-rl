from collections import deque
import random
import copy

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plot


def create_variables(network: tf.keras.Model, input_spec):
    pass


class QNetwork(tf.keras.Model):
    def __init__(self,
                 observation_shape,
                 action_size,
                 units=(24, 24),
                 name="QNetwork"):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(units[0],
                                            activation="relu",
                                            kernel_initializer="he_uniform")
        self.dense2 = tf.keras.layers.Dense(units[1],
                                            activation="relu",
                                            kernel_initializer="he_uniform")
        self.dense3 = tf.keras.layers.Dense(action_size,
                                            activation="linear",
                                            kernel_initializer="he_uniform")

        with tf.device("/cpu:0"):
            self(inputs=tf.constant(
                np.zeros(shape=(1, ) + observation_shape, dtype=np.float32)))

    @tf.function
    def call(self, inputs):
        features = self.dense1(inputs)
        features = self.dense2(features)
        q_values = self.dense3(features)
        return q_values


class DQN:
    def __init__(self,
                 observation_shape,
                 action_size,
                 q_network,
                 optimizer,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.999,
                 discount=0.99,
                 n_warmup=1000,
                 target_q_network=None,
                 target_update_period=100,
                 name="DQN"):
        self._action_size = action_size
        self._observation_ndim = np.array(observation_shape).shape[0]
        self._q_network: tf.keras.Model = q_network
        if target_q_network is not None:
            self._target_q_network = target_q_network
        else:
            self._target_q_network: tf.keras.Model = copy.deepcopy(q_network)
        self._optimizer = optimizer
        self._n_warmup = n_warmup

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        self._discount = discount

        self._target_update_period = target_update_period
        self._n_updates = 0
        self._name = name

    def select_action(self, observation, training=True):
        assert isinstance(observation, np.ndarray)
        if training and np.random.rand() < self._epsilon:
            action = np.random.randint(self._action_size)
            return action

        observations = np.expand_dims(observation, axis=0).astype(np.float32)
        actions = self._select_action_batch(observations)
        return actions.numpy()[0]

    @tf.function
    def _select_action_batch(self, observations):
        q_values = self._q_network(observations)
        return tf.argmax(q_values, axis=1)

    def train(self,
              observations,
              actions,
              next_observations,
              rewards,
              dones,
              weights=None):
        with tf.GradientTape() as tape:
            td_error, td_loss = self._loss(observations,
                                           actions,
                                           next_observations,
                                           rewards,
                                           dones,
                                           weights=weights,
                                           training=True)
            vars_to_train = self._q_network.trainable_weights
            grads = tape.gradient(td_loss, vars_to_train)
            self._optimizer.apply_gradients(zip(grads, vars_to_train))
        tf.summary.scalar(name=self._name + "/td_loss", data=td_loss)
        tf.summary.scalar(name=self._name + "/td_error", data=td_error)
        self._n_updates += 1
        # print("n_update: {}, td_error: {}, td_loss: {}".format(
        #     self._n_update, td_error, td_loss))
        if self._n_updates % self._target_update_period == 0:
            print("updating target q network ...")
            self._target_q_network.set_weights(self._q_network.get_weights())

        if self._epsilon > self._epsilon_min:
            self._epsilon = max(self._epsilon * self._epsilon_decay,
                                self._epsilon_min)
        tf.summary.scalar(name=self._name + "/epsilon", data=self._epsilon)
        return td_error, td_loss

    # @tf.function
    def _loss(self,
              observations,
              actions,
              next_observations,
              rewards,
              dones,
              weights=None,
              training=False):
        batch_size = observations.shape[0]
        not_dones = 1. - tf.cast(dones, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)

        # indices = tf.concat(values=(tf.expand_dims(tf.range(batch_size),
        #                                            axis=1), actions),
        #                     axis=1)
        # q_values = tf.expand_dims(tf.gather_nd(
        #     self._q_network(observations, training=training), indices),
        #                           axis=1)
        actions = tf.one_hot(actions,
                             self._action_size,
                             on_value=1.,
                             off_value=0.)
        q_values = tf.reduce_sum(
            self._q_network(observations, training=training) * actions, axis=1)
        next_q_values = tf.reduce_max(
            self._target_q_network(next_observations), axis=1)
        target_q_values = \
            tf.stop_gradient(rewards + self._discount * not_dones * next_q_values)

        td_error = tf.reduce_mean(target_q_values - q_values)
        td_loss = tf.losses.huber(target_q_values, q_values)
        if weights is not None:
            if not isinstance(weights, tf.Tensor):
                weights = tf.convert_to_tensor(weights, dtype=tf.float32)
            td_loss = td_loss * weights
        td_loss = tf.reduce_mean(td_loss)
        return td_error, td_loss


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)

    def add(self, observation, action, next_observation, reward, done):
        self.memory.append(
            (observation, action, next_observation, reward, done))

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


def train(env, agent, replay_buffer, batch_size=64):
    if len(replay_buffer) < agent._n_warmup:
        return
    batch_size = min(batch_size, len(replay_buffer))
    batch = replay_buffer.sample(batch_size)
    observations = np.array([x[0] for x in batch], dtype='float32')
    actions = np.array([x[1] for x in batch], dtype='int32')
    next_observations = np.array([x[2] for x in batch], dtype='float32')
    rewards = np.array([x[3] for x in batch], dtype='float32')
    dones = np.array([x[4] for x in batch], dtype='bool')
    td_error, td_loss = agent.train(observations, actions, next_observations,
                                    rewards, dones)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    q_network = QNetwork(observation_shape, action_size)
    q_network.summary()
    target_q_network = QNetwork(observation_shape, action_size)
    # target_q_network = None
    optimizer = tf.keras.optimizers.Adam(1e-3)
    agent = DQN(observation_shape,
                action_size,
                q_network,
                optimizer,
                target_q_network=target_q_network)
    replay_buffer = ReplayBuffer()

    EPISODES = 300

    scores, episodes = [], []
    e = 0

    while agent._n_updates < 1000000:
        done = False
        score = 0
        obs = env.reset()

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            reward = reward if not done or score == 499 else -100
            replay_buffer.add(obs, action, next_obs, reward, done)

            train(env, agent, replay_buffer)
            score += reward
            obs = next_obs

            if done:
                e += 1
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                plot.plot(episodes, scores, 'b')
                plot.savefig("./cartpole_dqn.png")
                print(
                    "episode: {}, score: {}, replay_buffer length: {}, epsilon: {}"
                    .format(e, score, len(replay_buffer), agent._epsilon))
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    import sys
                    sys.exit()
