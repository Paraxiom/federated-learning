# Federated Learning Simulation

## Project Overview
This project implements a federated learning simulation with a focus on Deep Reinforcement Learning (DRL) for optimizing task scheduling across distributed nodes. It uses TensorFlow to build and train deep learning models and evaluates their performance in a simulated environment.

## Features
- **Deep Q-Networks (DQN)**: Implementation of DQN for action decision-making in task scheduling.
- **Environment Simulation**: Custom environment for simulating node and task interactions.
- **Performance Metrics**: Evaluation of models based on task delay and resource utilization.
- **Visualization**: Graphical representation of training progress and results.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip3
- TensorFlow 2.x
- NumPy
- Matplotlib

### Setup
Clone the repository to your local machine:
```
git clone https://github.com/yourusername/federated-learning.git
cd federated-learning
pip install -r requirements.txt
python3 __main__.py
python -m unittest discover
```