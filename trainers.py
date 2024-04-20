import numpy as np
import tensorflow as tf
from models import DQN  # Make sure to correctly import your DQN class
import logging

# In trainers.py
class DeepRMTrainer:
    def __init__(self, environment, episodes=300, save_interval=10, copy_steps=32):
        self.environment = environment
        self.episodes = episodes
        self.save_interval = save_interval
        self.copy_steps = copy_steps
        self.epsilon = 0.99
        self.min_epsilon = 0.1
        input_shape = (environment.summary().shape[0], environment.summary().shape[1], 1)
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.dqn_train = DQN(input_shape, output_shape)
        self.dqn_target = DQN(input_shape, output_shape)


    def train(self, agents=None):  # Optional agents parameter for federated scenarios
        total_rewards = np.empty(self.episodes)
        for episode in range(self.episodes):
            total_rewards[episode] = self.train_episode(agents)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            if episode % self.save_interval == 0:
                self.dqn_target.save_weights()
            logging.info(f"Episode {episode}: Total Reward: {total_rewards[episode]}")
        return total_rewards

    def train_episode(self, agents):
        total_reward = 0
        self.environment.reset()
        while not self.environment.terminated():
            current_state = self.environment.get_state().reshape((1, -1))  # Example reshape for a 1D state into 2D

            action = self.dqn_train.get_action(current_state, self.epsilon)
            reward, next_state, done = self.environment.step(action)
            self.dqn_train.add_experience(current_state, action, reward, next_state, done)
            self.dqn_train.train(self.dqn_target)
            total_reward += reward
            if self.environment.current_step % self.copy_steps == 0:
                self.dqn_target.copy_weights(self.dqn_train)
            if agents:  # Handle federated or multi-agent scenarios
                for agent in agents:
                    agent.update_model()  # Adjust as necessary
            return total_reward

    def average_parameters(self, dqn_list):
        num_dqns = len(dqn_list)
        average_weights = []
        for layer_index in range(len(dqn_list[0].model.trainable_variables)):
            layer_weights = np.mean([dqn.model.trainable_variables[layer_index].numpy() for dqn in dqn_list], axis=0)
            average_weights.append(layer_weights)
        for dqn in dqn_list:
            for layer_index, layer in enumerate(dqn.model.trainable_variables):
                layer.assign(average_weights[layer_index])
