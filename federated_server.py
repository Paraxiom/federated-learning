import numpy as np

class Server:
    def __init__(self, model):
        # Initialize server with a global model
        self.global_model = model

    def update_global_model(self, new_weights):
        # Update global model weights
        self.global_model.set_weights(new_weights)

class Agent:
    def __init__(self, environment, model):
        self.environment = environment
        self.model = model
    def decide_action(self):
        current_state = self.environment.get_current_state()  # Make sure this method exists and is correct
        action = self.model.get_action(current_state)
        return action

def federate_and_update_agents(agents, server):
    # Collect weights from all agents
    local_weights = [agent.model.get_weights() for agent in agents]
    # Average the weights
    averaged_weights = np.mean(local_weights, axis=0)
    # Update the server's global model
    server.update_global_model(averaged_weights)
    # Distribute the updated weights back to agents
    for agent in agents:
        agent.model.set_weights(averaged_weights)
