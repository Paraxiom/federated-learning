import os
import matplotlib.pyplot as plt

def node_plot(allActions):
    if not os.path.exists('__cache__/node_plot'):
        os.makedirs('__cache__/node_plot')
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
    plt.xlabel('Nodes')
    plt.ylabel('Number of Scheduled Tasks')
    plt.title('Scheduled Tasks in Each Node')
    # Save the plot to a file
    plt.savefig("__cache__/node_plot/Scheduled_Tasks.png")

    # Show the plot (if you want to display it)
    #plt.show()


'''
import numpy as np
def reward_plot(all_total_rewards):
    if not os.path.exists('__cache__/reward_plot'):
        os.makedirs('__cache__/reward_plot')
    for i in range(len(all_total_rewards)):
        all_total_rewards[i] = all_total_rewards[i]

    plt.plot(range(0, len(all_total_rewards)), all_total_rewards, marker='o')
    plt.xlabel("Episode")  # Modify x-axis label
    plt.ylabel("Task Slowdown")

    # Save the plot to a file
    plt.savefig("__cache__/reward_plot/all_reward.png")

    # Show the plot (if you want to display it)
    plt.show()


def d_rewards_plot(discounted_rewards):
    if not os.path.exists('__cache__/reward_plot'):
        os.makedirs('__cache__/reward_plot')
    # Calculate discounted rewards after collecting all rewards in the episode
    for i in range(len(discounted_rewards)):
        discounted_rewards[i] = discounted_rewards[i]

    plt.plot(range(0, len(discounted_rewards)), discounted_rewards, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Discounted_Total_Rewards")
    plt.savefig("__cache__/reward_plot/Discounted_Reward_Evolution_Over_Episodes.png")
    plt.show()
'''
'''
def reward_plots(all_r):
    if not os.path.exists('__cache__/reward_plot'):
        os.makedirs('__cache__/reward_plot')
    for i in range(len(all_r)):
        all_r[i] = all_r[i]

    plt.plot(range(0, len(all_r)), all_r, marker='o')
    plt.xlabel("Episode")  # Modify x-axis label
    plt.ylabel("Total reward-reset")
    plt.savefig("__cache__/reward_plot/Reward Evolution Over Episode.png")
    plt.show()
'''

'''

def d_rewards_plot(discounted_rewards):
    if not os.path.exists('__cache__/reward_plot'):
        os.makedirs('__cache__/reward_plot')
    for i in range(len(discounted_rewards)):
        # Calculate discounted rewards after collecting all rewards in the episode
        discounted_rewards[i] += 0.99 * discounted_rewards[i - 1]

    plt.plot(range(0, len(discounted_rewards)), discounted_rewards, marker='o')
    plt.xlabel("Episodes")  # Modify x-axis label
    plt.ylabel("Discounted_Rewards")
    plt.savefig("__cache__/reward_plot/Discounted_Reward Evolution Over Episodes.png")
    plt.show()


def reward_plots(all_r):
    for i in range(len(all_r)):
        all_r[i] = all_r[i]

    plt.plot(range(0, len(all_r)), all_r, marker='o')
    plt.xlabel("Iterations")  # Modify x-axis label
    plt.ylabel("Rewards")
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


'''

# this is reward for each itration in each epesode 1000 itration in each epesode

'''


# this is plot nodes with tasks without numbers, showing the scheduling time in each node

def node_plots(allActions):
    node1 = []
    node2 = []
    node3 = []
    node4 = []
    node5 = []

    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    for i in range(len(allActions)):

        if allActions[i].node.label == 'node1':
            node1.append(allActions[i].task.label)
            y1.append(i)
        elif allActions[i].node.label == 'node2':
            node2.append(allActions[i].task.label)
            y2.append(i)
        elif allActions[i].node.label == 'node3':
            node3.append(allActions[i].task.label)
            y3.append(i)
        elif allActions[i].node.label == 'node4':
            node4.append(allActions[i].task.label)
            y4.append(i)
        elif allActions[i].node.label == 'node5':
            node5.append(allActions[i].task.label)
            y5.append(i)

        print(allActions[i].node.label)

    #node1
    x1 = [1] * len(node1)
    plt.scatter(x1, y1)

    #node2
    x2 = [2] * len(node2)
    plt.scatter(x2, y2)

    #node3
    x3 = [3] * len(node3)
    plt.scatter(x3, y3)
    # node4
    x4 = [4] * len(node4)
    plt.scatter(x4, y4)
    # node5
    x5 = [5] * len(node5)
    plt.scatter(x5, y5)

    plt.title("Task Scheduling on Edge Nodes")  # Add title
    plt.xlabel("Tasks")  # Modify x-axis label
    plt.ylabel("Nodes")  # Modify y-axis label

    plt.show()
    print()'''
