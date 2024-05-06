import unittest
from env import Environment, load
from node import Node
from task import Task
from schedule import DQN
from FL_AVG import combine_agents, distribute_agents

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # Define nodes and initialize Environment
        nodes = [Node([2, 2], 10, "Node1"), Node([2, 2], 10, "Node2")]
        tasks = [Task([1, 1], 2, "Task1"), Task([1, 1], 3, "Task2")]
        self.environment = Environment(nodes, 10, 20, task_generator=iter(tasks))

        # Define model shapes and actions for DQN
        self.model_input_shape = (2, 2, 1)  # Correct shape for Conv2D input
        self.output_shape = 10  # Number of outputs
        self.num_actions = 2  # Number of actions

        # Initialize two DQN models for testing
        self.dqn1 = DQN(self.model_input_shape, self.output_shape, self.num_actions)
        self.dqn2 = DQN(self.model_input_shape, self.output_shape, self.num_actions)

    def test_initialization(self):
        """Check initial setup of the environment."""
        self.assertEqual(len(self.environment.nodes), 2)
        self.assertEqual(self.environment.queue_size, 10)
        self.assertEqual(self.environment.backlog_size, 20)

    def test_step_function_no_action(self):
        """Verify that no action results in expected state changes."""
        initial_reward, initial_state, initial_done = self.environment.step(self.environment.get_number_of_actions() - 1)
        self.assertTrue(initial_reward <= 0, "Reward should be zero or negative for no action")
        self.assertFalse(initial_done, "Environment should not terminate on no action")
        self.assertIsNotNone(initial_state, "Next state should still be provided")

    def test_task_scheduling(self):
        """Ensure task scheduling updates queue and node state."""
        # Add a task to the environment's queue directly for controlled testing
        task_to_schedule = Task([1, 1], 2, "Task1")
        self.environment.queue.append(task_to_schedule)
        initial_task_count = len(self.environment.queue)
        initial_node_task_count = sum(len(node.scheduled_tasks) for node in self.environment.nodes)

        # Attempt to schedule the first task in the queue
        self.environment.step(0)  # Assuming 0 targets the first task for the first node

        # Check if the task count in the queue decreased
        self.assertEqual(len(self.environment.queue), initial_task_count - 1, "Queue should decrease by 1")

        # Optionally, check if the task count on the node increased (if your environment tracks this)
        new_node_task_count = sum(len(node.scheduled_tasks) for node in self.environment.nodes)
        self.assertEqual(new_node_task_count, initial_node_task_count + 1, "Node task count should increase by 1")

        # Verify that the task scheduled is the one intended (if applicable)
        # This part depends on your environment's implementation details


class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        # Define model input shape, output shape, and actions
        self.model_input_shape = (4, 4, 1)  # Adjusted shape to match Conv2D requirements
        self.output_shape = 2  # Adjusted based on actual network configuration
        self.num_actions = 2

        # Initialize two DQN models for federated learning tests
        self.dqn1 = DQN(self.model_input_shape, self.output_shape, self.num_actions)
        self.dqn2 = DQN(self.model_input_shape, self.output_shape, self.num_actions)

    def test_combine_agents(self):
        """Test if combining agents properly averages their parameters."""
        combined_dqn = combine_agents(self.dqn1, [self.dqn1, self.dqn2], [0.5, 0.5])
        # Assertions to verify that parameters have been averaged should be implemented here

    def test_distribute_agents(self):
        """Ensure that distributing weights updates all agents equally."""
        distribute_agents(self.dqn1, [self.dqn1, self.dqn2])
        # Assertions to verify that weights are equally distributed should be implemented here

if __name__ == '__main__':
    unittest.main()
