from schedule import *

# Adjusting combine_agents in FL_AVG.py to average based on scores or data volume
def combine_agents(main_agent, agents, scores):
    total_score = sum(scores)
    if total_score == 0:
        print("Warning: Total score is zero. Agents will not be updated.")
        return main_agent  # or handle this case differently based on your needs

    for i in range(len(agents)):
        for main_param, agent_param in zip(main_agent.dqn_train.model.trainable_variables, agents[i].dqn_train.model.trainable_variables):
            if i == 0:
                main_param.assign(agent_param * (scores[i] / total_score))
            else:
                main_param.assign_add(agent_param * (scores[i] / total_score))

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
    for i in range(len(agents)):
        for main_agent_param, agent_param in zip(main_agent.dqn_target.model.trainable_variables,  agents[i].dqn_target.model.trainable_variables):
            agent_param.assign(main_agent_param)
        for main_agent_param, agent_param in zip(main_agent.dqn_train.model.trainable_variables, agents[i].dqn_train.model.trainable_variables):
            agent_param.assign(main_agent_param)
    return agents

