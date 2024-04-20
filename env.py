"""Environment Manipulation."""

import json
import os
import random

import numpy as np
from PIL import Image

from node import Node
from task import Task
from schedule import CompactScheduler
from schedule import SpreadScheduler
from schedule import DeepRMScheduler


class Environment:
    """Environment Simulation."""

    def __init__(self, nodes, queue_size, backlog_size, task_generator):
        self.nodes = nodes
        self.queue_size = queue_size
        self.backlog_size = backlog_size
        self.queue = []
        self.backlog = []
        self.timestep_counter = 0
        self._task_generator = task_generator
        self._task_generator_end = False
    
    def reset(self):
        # Reset environment to initial conditions
        self.queue = []
        self.backlog = []
        self.timestep_counter = 0
        # Possibly other state initializations
        # Return the initial state of the environment if necessary
        return self.summary()
    
    def get_state(self):
        # Collect relevant state information
        state = np.array([node.utilization() for node in self.nodes] + [len(self.queue), len(self.backlog)])
        # Calculate the total size needed to match the reshape target
        total_size = 10 * 10 * 3  # This is 300 for a (10, 10, 3) shape
        # Calculate the padding size needed
        padding_size = total_size - len(state)
        if padding_size > 0:
            # Pad the state array if it's less than required
            state = np.pad(state, (0, padding_size), mode='constant', constant_values=0)
        else:
            # Or slice the state array if it's more than required
            state = state[:total_size]
        # Reshape to fit the model input shape
        return state.reshape((10, 10, 3))
    
    def get_current_state(self):
        # Assuming you have a way to define what the current state is
        return self.current_state

    def step(self, action):
        # Example: Process the action and update the environment
        # This is a simplified example. You'll need to adapt this to fit your environment's logic.
        reward = 0
        done = False

        # Example action processing logic
        if action == 'move':
            # Update environment state
            reward = 1  # Assign reward based on action's outcome
            if self.some_end_condition:
                done = True

        # Assuming self.summary() returns the current state of the environment
        new_state = self.summary()
        
        return new_state, reward, done

    def timestep(self):
        """Proceed to the next timestep."""
        self.timestep_counter += 1

        # each node proceeds to the next timestep
        for node in self.nodes:
            node.timestep()

        # move tasks from backlog to queue
        p_queue = len(self.queue)
        p_backlog = 0
        indices = []
        while p_queue < self.queue_size and p_backlog < len(self.backlog):
            self.queue.append(self.backlog[p_backlog])
            indices.append(p_backlog)
            p_queue += 1
            p_backlog += 1
        for i in sorted(indices, reverse=True):
            del self.backlog[i]

        # accept more tasks, move to backlog
        p_backlog = len(self.backlog)
        while p_backlog < self.backlog_size:
            new_task = next(self._task_generator, None)
            if new_task is None:
                self._task_generator_end = True
                break
            else:
                self.backlog.append(new_task)
                p_backlog += 1

    def terminated(self):
        """Check environment termination."""
        for node in self.nodes:
            if node.utilization() > 0:
                return False
        if self.queue or self.backlog or not self._task_generator_end:
            return False
        return True

    def reward(self):
        """Reward calculation."""
        r = 0
        for node in self.nodes:
            if node.scheduled_tasks:
                r += 1/sum([task[0].duration for task in node.scheduled_tasks])
        if self.queue:
            r += 1/sum([task.duration for task in self.queue])
        if self.backlog:
            r += 1/sum([task.duration for task in self.backlog])
        return -r

    def summary(self, bg_shape=None):
        """State representation."""
        # background shape
        if bg_shape is None:
            bg_col = max([max(node.resources) for node in self.nodes])
            bg_row = max([node.duration for node in self.nodes])
            bg_shape = (bg_row, bg_col)

        if len(self.nodes) > 0:
            dimension = self.nodes[0].dimension

            # state of nodes
            temp = self.nodes[0].summary(bg_shape)
            for i in range(1, len(self.nodes)):
                temp = np.concatenate((temp, self.nodes[i].summary(bg_shape)), axis=1)

            # state of occupied queue slots
            for i in range(len(self.queue)):
                temp = np.concatenate((temp, self.queue[i].summary(bg_shape)), axis=1)

            # state of vacant queue slots
            empty_summary = Task([0]*dimension, 0, 'empty_task').summary(bg_shape)
            for i in range(len(self.queue), self.queue_size):
                temp = np.concatenate((temp, empty_summary), axis=1)

            # state of backlog
            backlog_summary = Task([0], 0, 'empty_task').summary(bg_shape)
            p_backlog = 0
            p_row = 0
            p_col = 0
            while p_row < bg_shape[0] and p_col < bg_shape[1] and p_backlog < len(self.backlog):
                backlog_summary[p_row, p_col] = 0
                p_row += 1
                if p_row == bg_shape[0]:
                    p_row = 0
                    p_col += 1
                p_backlog += 1
            temp = np.concatenate((temp, backlog_summary), axis=1)

            return temp
        else:
            return None

    def plot(self, bg_shape=None):
        """Plot state representation into image."""
        if not os.path.exists('__cache__/state'):
            os.makedirs('__cache__/state')
        summary_matrix = self.summary(bg_shape)
        summary_plot = np.full((summary_matrix.shape[0], summary_matrix.shape[1]), 255, dtype=np.uint8)
        for row in range(summary_matrix.shape[0]):
            for col in range(summary_matrix.shape[1]):
                summary_plot[row, col] = summary_matrix[row, col]
        Image.fromarray(summary_plot).save('__cache__/state/environment_{0}.png'.format(self.timestep_counter))

    def __repr__(self):
        return 'Environment(timestep_counter={0}, nodes={1}, queue={2}, backlog={3})'.format(self.timestep_counter, self.nodes, self.queue, self.backlog)


def load(load_environment=True, load_scheduler=True):
    """Load environment and scheduler from conf/env.conf.json"""
    tasks = _load_tasks()
    task_generator = (t for t in tasks)
    with open('conf/env.conf.json', 'r') as fr:
        data = json.load(fr)
        nodes = []
        label= 0
        for node_json in data['nodes']:
            label += 1
            nodes.append(Node(node_json['resource_capacity'], node_json['duration_capacity'], 'node' + str(label)))
        environment = None
        scheduler = None
        if load_environment:
            environment = Environment(nodes, data['queue_size'], data['backlog_size'], task_generator)
            environment.timestep()
        if load_scheduler:
            if 'CompactScheduler' == data['scheduler']:
                scheduler = CompactScheduler(environment)
            elif 'SpreadScheduler' == data['scheduler']:
                scheduler = SpreadScheduler(environment)
            else:
                scheduler = DeepRMScheduler(environment, data['train'])
        return (environment, scheduler)


def _load_tasks():
    """Load tasks from __cache__/tasks.csv"""
    _generate_tasks()
    tasks = []
    with open('__cache__/tasks.csv', 'r') as fr:
        resource_indices = []
        duration_index = 0
        label_index = 0
        line = fr.readline()
        parts = line.strip().split(',')
        for i in range(len(parts)):
            if parts[i].strip().startswith('resource'):
                resource_indices.append(i)
            if parts[i].strip() == 'duration':
                duration_index = i
            if parts[i].strip() == 'label':
                label_index = i
        line = fr.readline()
        while line:
            parts = line.strip().split(',')
            resources = []
            for index in resource_indices:
                resources.append(int(parts[index]))
            tasks.append(Task(resources, int(parts[duration_index]), parts[label_index]))
            line = fr.readline()
    return tasks


def _generate_tasks():
    """Generate tasks according to conf/task.pattern.conf.json"""
    if not os.path.exists('__cache__'):
        os.makedirs('__cache__')
    if os.path.isfile('__cache__/tasks.csv'):
        return
    with open('conf/task.pattern.conf.json', 'r') as fr, open('__cache__/tasks.csv', 'w') as fw:
        data = json.load(fr)
        if len(data) > 0:
            for i in range(len(data[0]['resource_range'])):
                fw.write('resource' + str(i+1) + ',')
            fw.write('duration,label' + '\n')
        label = 0
        for task_pattern in data:
            for i in range(task_pattern['batch_size']):
                label += 1
                resources = []
                duration = str(random.randint(task_pattern['duration_range']['lowerLimit'], task_pattern['duration_range']['upperLimit']))
                for j in range(len(task_pattern['resource_range'])):
                    resources.append(str(random.randint(task_pattern['resource_range'][j]['lowerLimit'], task_pattern['resource_range'][j]['upperLimit'])))
                fw.write(','.join(resources) + ',' + duration +  ',' + 'task' + str(label) + '\n')
