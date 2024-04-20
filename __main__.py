from models import DQN
from federated_server import Agent, Server, federate_and_update_agents
import env

if __name__ == '__main__':
    print('Start---')

    # Setup environments and agents
    number_of_agents = 5
    input_shape = (10, 10, 3)  # Define the input shape
    number_of_actions = 5       # Define the number of actions

    environments, schedulers = zip(*[env.load() for _ in range(number_of_agents)])
    models = [DQN(input_shape, number_of_actions) for _ in range(number_of_agents)]
    agents = [Agent(environment, model) for environment, model in zip(environments, models)]
    global_model = DQN(input_shape, number_of_actions)
    server = Server(global_model)

    # Main simulation loop
    number_of_training_cycles = 100  # Define the number of training cycles
    for cycle in range(number_of_training_cycles):
        for agent, scheduler in zip(agents, schedulers):
            while not agent.environment.terminated():
                current_state = agent.environment.get_state()  # Use the correct method to get the current state
                action = agent.model.get_action(current_state)  # Decision-making by the agent
                new_state, reward, done = agent.environment.step(action)  # Environment processes the action
                agent.model.add_experience(current_state, action, reward, new_state, done)  # Update model with new experience
                if done:
                    break
            agent.model.train()  # Local model updates after the episode

        federate_and_update_agents(agents, server)  # Federated averaging after all agents update

    print('End---')
