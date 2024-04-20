"""Schedulers."""
# In schedule.py
from models import DQN

import datetime
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
import env
import logging
from trainers import DeepRMTrainer


class Action(object):
    """Schedule Action."""

    def __init__(self, task, node):
        self.task = task
        self.node = node

    def __repr__(self):
        return 'Action(task={0} -> node={1})'.format(self.task.label, self.node.label)


class Scheduler(ABC):
    """Scheduler Interface."""

    @abstractmethod
    def schedule(self):
        pass


class CompactScheduler(Scheduler):
    """Compact scheduler."""

    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        """Higher priority for higher utilization."""
        actions = []
        indices = []

        # sort nodes according to reversed utilization, schedule tasks from queue to nodes
        for i_task in range(len(self.environment.queue)):
            pairs = [(i_node, self.environment.nodes[i_node].utilization()) for i_node in range(len(self.environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
            for pair in pairs:
                if self.environment.nodes[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.nodes[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]

        # proceed to the next timestep
        self.environment.timestep()

        return actions


class SpreadScheduler(Scheduler):
    """Spread scheduler."""

    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        """Higher priority for lower utilization."""
        actions = []
        indices = []

        # sort nodes according to utilization, schedule tasks from queue to nodes
        for i_task in range(len(self.environment.queue)):
            pairs = [(i_node, self.environment.nodes[i_node].utilization()) for i_node in range(len(self.environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1])
            for pair in pairs:
                if self.environment.nodes[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.nodes[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]

        # proceed to the next timestep
        self.environment.timestep()

        return actions






class DeepRMScheduler(Scheduler):
    """DeepRM scheduler."""

    def __init__(self, environment, train=False):
        print(type(environment), hasattr(environment, 'summary'))
        self.environment = environment
        if train:
            self.trainer = DeepRMTrainer(environment)
        else:
            self.trainer = None

    def train(self, agents):
        if self.trainer:
            self.trainer.train(agents)

    def schedule(self):
        """Schedule with trained model."""
        actions = []

        # apply actions until there's an invalid one
        while True:
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, epsilon=0.1)  # Set epsilon to a low non-zero value
            task_index, node_index = self._explain(action_index)
            if task_index < 0 or node_index < 0:
               break
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                break
            del self.environment.queue[task_index]
            actions.append(Action(scheduled_task, scheduled_node))

        # proceed to the next timestep
        self.environment.timestep()

        return actions

    def _explain(self, action_index):
        """Explain action."""
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit*node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return (task_index, node_index)
