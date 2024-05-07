from schedule import *

# Adjusting combine_agents in FL_AVG.py to average based on scores or data volume
def combine_agents(main_agent, agents, scores):
    total_score = sum(scores)
    if total_score == 0:
        print("Warning: Total score is zero. Agents will not be updated.")
        return main_agent

    combined_weights = None
    for i, agent in enumerate(agents):
        # Check if the agent is a scheduler or a direct DQN instance
        if isinstance(agent, DeepRMScheduler):
            agent_weights = agent.dqn_train.model.get_weights()
        elif isinstance(agent, DQN):
            agent_weights = agent.model.get_weights()
        else:
            continue  # or raise an error

        if combined_weights is None:
            combined_weights = [weight * (scores[i] / total_score) for weight in agent_weights]
        else:
            for j in range(len(combined_weights)):
                combined_weights[j] += agent_weights[j] * (scores[i] / total_score)

    if isinstance(main_agent, DeepRMScheduler):
        main_agent.dqn_train.model.set_weights(combined_weights)
    elif isinstance(main_agent, DQN):
        main_agent.model.set_weights(combined_weights)

    return main_agent



def combine_agents_reward_based(main_agent, agents, scores):
    total_reward = sum(scores)

    for i in range(len(agents)):
        for main_param, agent_param in zip(main_agent.dqn_train.model.trainable_variables, agents[i].dqn_train.model.trainable_variables):
            if i == 0:
                main_param.assign(agent_param * (scores[i] / total_reward))
            else:
                main_param.assign(main_param + agent_param * (scores[i] / total_reward))

        for main_param, agent_param in zip(main_agent.dqn_target.model.trainable_variables, agents[i].dqn_target.model.trainable_variables):
            if i == 0:
                main_param.assign(agent_param * (scores[i] / total_reward))
            else:
                main_param.assign(main_param + agent_param * (scores[i] / total_reward))

    return main_agent

def distribute_agents(main_agent, agents):
    main_weights = main_agent.model_train.get_weights()  # Assuming model_train is correct
    for agent in agents:
        agent.model_train.set_weights(main_weights)
    return agents
