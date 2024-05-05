"""Entrance."""
import env
from schedule import DeepRMScheduler as SchedulerTrain
from FL_AVG import combine_agents, distribute_agents
import pandas as pd
from plot import node_plot

def test_agent(test_environment, dqn_scheduler, runs_test=3):
    scores = []
    for i in range(runs_test):
        actions, score = dqn_scheduler.schedule()
        print(f'run_test number {i}', score)
        scores.append(score)
    return sum(scores) / len(scores)

if __name__ == '__main__':
    print('Start---')

    number_of_agents = 3
    environments, _ = zip(*[env.load() for _ in range(number_of_agents)])
    dqn_schedulers = [SchedulerTrain(environment) for environment in environments]

    global_environment, global_dqn_scheduler = env.load()
    single_environment, single_dqn_scheduler = env.load()

    scores_single_agent = []
    scores_global_agent = []
    
    # Aggregation and training loop
    for run in range(5):
        # Train each agent's DQN scheduler and collect scores
        local_scores = []
        for i, scheduler in enumerate(dqn_schedulers):
            print(f'Start training DQN scheduler {i}:')
            scheduler.train()
            score = test_agent(environments[i], scheduler)
            local_scores.append(score)
            print(f'The score of DQN scheduler {i}:', score)
        
        # Update global model using combined agents
        global_dqn_scheduler = combine_agents(global_dqn_scheduler, dqn_schedulers, local_scores)
        
        # Distribute the updated global model to each agent
        dqn_schedulers = distribute_agents(global_dqn_scheduler, dqn_schedulers)
        
        # Test and collect scores for both single and global DQN schedulers
        score_single = test_agent(single_environment, single_dqn_scheduler)
        scores_single_agent.append(score_single)
        
        score_global = test_agent(global_environment, global_dqn_scheduler)
        scores_global_agent.append(score_global)
        
        print(f"Run {run}: Single DQN Score: {score_single}, Global DQN Score: {score_global}")
    
    # Save the scores to CSV files
    pd.DataFrame({
        'Single Agent': scores_single_agent,
        'Global Agent': scores_global_agent
    }).to_csv('agent_scores_comparison.csv', index=False)


def main_training_loop():
    environments, schedulers = initialize_environments_and_schedulers(num_agents)
    global_scheduler = initialize_global_scheduler()
    
    for episode in range(total_episodes):
        local_scores = []
        
        # Train each local scheduler and compute scores
        for env, scheduler in zip(environments, schedulers):
            train_scheduler(env, scheduler)
            score = evaluate_scheduler(env, scheduler)
            local_scores.append(score)
        
        # Federated averaging
        global_scheduler = combine_agents(global_scheduler, schedulers, local_scores)
        schedulers = distribute_agents(global_scheduler, schedulers)
        
        # Optionally evaluate global scheduler
        global_score = evaluate_scheduler(global_environment, global_scheduler)
        print(f'Episode {episode}: Global Score: {global_score}')

        # Save scores and models
        save_scores_and_models(schedulers, global_scheduler, episode)

def plot_results(scores_single_agent, scores_global_agent):
    plt.plot(scores_single_agent, label='Single Agent')
    plt.plot(scores_global_agent, label='Global Agent')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.savefig('comparison_plot.png')
    plt.show()

def save_scores(scores_single_agent, scores_global_agent):
    results = pd.DataFrame({
        'Single Agent': scores_single_agent,
        'Global Agent': scores_global_agent
    })
    results.to_csv('scores_comparison.csv', index=False)