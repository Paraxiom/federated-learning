import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input

def build_model(input_shape, number_of_actions):
    """Build and return a new Keras model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(number_of_actions, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class DQN:
    """Deep Q-Network implementation."""
    def __init__(self, input_shape, number_of_actions):
        self.model = build_model(input_shape, number_of_actions)
        self.epsilon = 0.1  # Initial epsilon for ε-greedy policy
        self.gamma = 0.99  # Discount factor
        self.min_experiences = 100  # Minimum experiences before training
        self.max_experiences = 10000  # Maximum experiences in memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

    def train(self):
        if len(self.experience['s']) < self.min_experiences:
            return
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=32)  # Sample batch size
        states = np.array(self.experience['s'])[ids]
        actions = np.array(self.experience['a'])[ids]
        rewards = np.array(self.experience['r'])[ids]
        states_next = np.array(self.experience['s2'])[ids]
        dones = np.array(self.experience['done'])[ids]
        values_next = np.max(self.model.predict(states_next), axis=1)
        target_values = np.where(dones, rewards, rewards + self.gamma * values_next)

        with tf.GradientTape() as tape:
            current_predictions = self.model(states)
            selected_action_values = tf.reduce_sum(
                current_predictions * tf.one_hot(actions, depth=current_predictions.shape[1]), axis=1)
            loss = tf.reduce_mean(tf.square(target_values - selected_action_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.model.output_shape[-1])
        else:
            predicted_actions = self.predict(state[np.newaxis])
            return np.argmax(predicted_actions)

def average_dqn_parameters(dqn_list):
    avg_weights = []
    for layer_weights in zip(*(dqn.model.get_weights() for dqn in dqn_list)):
        avg_weights.append(np.mean(layer_weights, axis=0))
    for dqn in dqn_list:
        dqn.model.set_weights(avg_weights)

def test_global_model(dqn, test_environment):
    total_reward = 0
    while not test_environment.terminated():
        state = test_environment.get_state()
        action = dqn.get_action(state, epsilon=0)  # Greedy action
        new_state, reward, done = test_environment.step(action)
        total_reward += reward
    return total_reward
