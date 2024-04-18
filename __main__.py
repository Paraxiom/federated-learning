"""Entrance."""

import env
from plot import node_plot

if __name__ == '__main__':
    print('start---')

    environment, scheduler = env.load() # here it will return already trained DQN agent.. the polt of reward show the agent learning will be plot at the end of everything
    # scheduler is the trained DQN agent

    all_actions = []

    while not environment.terminated():
       # environment.plot()
        actions = scheduler.schedule()

        # Print debugging information at each timestep
        print("Timestep:", environment.timestep_counter)

        # Print utilization for each node
        for node in environment.nodes:
            print(f"Node {node.label} utilization: {node.utilization()}")

        # Print the status of the queue
        print(f"Queue size: {len(environment.queue)}")

        # Print the status of the backlog
        print(f"Backlog size: {len(environment.backlog)}")

        # Check if the task generator has ended
        print("Task generator ended:", environment._task_generator_end)


        if len(actions) != 0:
            print(actions)
            all_actions += actions
        else:
            print('no scheduler actions !')

    #node_plot(all_actions)

    print('END')

