from FL_AVG import combine_agents, distribute_agents
from schedule import DQN

from schedule import DQN
import unittest
class TestFederatedLearning(unittest.TestCase):
    def setUp(self):
        self.model_input_shape = (2, 2, 1)  # Example shape
        self.output_shape = 10  # Example output size
        self.num_actions = 2
        self.dqn1 = DQN(self.model_input_shape, self.output_shape, self.num_actions)
        self.dqn2 = DQN(self.model_input_shape, self.output_shape, self.num_actions)
    def test_combine_agents(self):
        """ Test if combining agents properly averages their parameters. """
        combined_dqn = combine_agents(self.dqn1, [self.dqn1, self.dqn2], [0.5, 0.5])
        # You would add assertions here to check if the parameters are indeed averaged

    def test_distribute_agents(self):
        """ Test if distributing weights updates all agents equally. """
        distribute_agents(self.dqn1, [self.dqn1, self.dqn2])
        # Assertions to check if dqn1 and dqn2 now share the same weights

if __name__ == '__main__':
    unittest.main()
