from models import DQN
from federated_server import Agent, Server, federate_and_update_agents
import env
import matplotlib.pyplot as plt
import plot


def reward_plot(all_total_rewards, filename='all_reward.png', title='Rewards per Episode', xlabel='Episode', ylabel='Total Reward'):
    """
    Plot the total rewards over episodes.

    Parameters:
    - all_total_rewards: List of total rewards per episode.
    - filename: The name of the file to save the plot.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    if not os.path.exists('__cache__/reward_plot'):
        os.makedirs('__cache__/reward_plot')

    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(all_total_rewards, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  # Add a grid for easier visualization

    # Save the plot to a file in the specified directory
    plt.savefig(f'__cache__/reward_plot/{filename}')
    plt.close()  # Close the plot to free up memory

    # Optionally, you can include plt.show() if you want to display the plot interactively
    plt.show()


def node_plot(allActions):
        node_counts = {}
        for action in allActions:
            node_label = action.node.label
            if node_label in node_counts:
                node_counts[node_label] += 1
            else:
                node_counts[node_label] = 1

        nodes = list(node_counts.keys())
        counts = [node_counts[node] for node in nodes]

        plt.bar(nodes, counts)
        plt.xlabel('Node')
        plt.ylabel('Number of Scheduled Tasks')
        plt.title('Scheduled Tasks in Each Node')
        plt.show()
        print()



if __name__ == '__main__':
    print('Start---')

    # Setup environments and agents
    number_of_agents = 5
    input_shape = (10, 10, 3)  # Define the input shape
    number_of_actions = 5  # Define the number of actions

    environments, schedulers = zip(*[env.load() for _ in range(number_of_agents)])
    models = [DQN(input_shape, number_of_actions) for _ in range(number_of_agents)]
    agents = [Agent(environment, model) for environment, model in zip(environments, models)]
    global_model = DQN(input_shape, number_of_actions)
    server = Server(global_model)

    rewards = []  # Store total rewards from each episode
    allActions = []  # Assuming you collect actions somewhere in your loop

    # Main simulation loop
    number_of_training_cycles = 100  # Define the number of training cycles
    for cycle in range(number_of_training_cycles):
        episode_rewards = 0  # Track rewards per episode
        for agent, scheduler in zip(agents, schedulers):
            while not agent.environment.terminated():
                current_state = agent.environment.get_state()
                action = agent.model.get_action(current_state)
                new_state, reward, done = agent.environment.step(action)
                agent.model.add_experience(current_state, action, reward, new_state, done)
                episode_rewards += reward
                allActions.append(action)  # You need to define how actions are stored
                if done:
                    break
            agent.model.train()

        federate_and_update_agents(agents, server)  # Federated averaging
        rewards.append(episode_rewards)  # Append total rewards for the episode

    reward_plot(rewards, filename='episode_rewards.png', title='Episode Rewards Overview')
    node_plot(allActions)

    print('End---')






