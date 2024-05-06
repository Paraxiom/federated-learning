import matplotlib.pyplot as plt
import pandas as pd
import env
from schedule import DeepRMScheduler as SchedulerTrain
from FL_AVG import combine_agents, distribute_agents

def test_agent(environment, scheduler, runs_test=3):
    total_scores = []
    for i in range(runs_test):
        actions, run_scores = scheduler.schedule()
        average_score = sum(run_scores) / len(run_scores) if run_scores else 0
        total_scores.append(average_score)
    return sum(total_scores) / len(total_scores) if total_scores else 0

def main():
    print('Start---')

    number_of_agents = 3
    environments, _ = zip(*[env.load() for _ in range(number_of_agents)])
    dqn_schedulers = [SchedulerTrain(environment) for environment in environments]

    global_environment, global_dqn_scheduler = env.load()
    single_environment, single_dqn_scheduler = env.load()

    scores_single_agent = []
    scores_global_agent = []

    for run in range(5):
        local_scores = []
        for i, scheduler in enumerate(dqn_schedulers):
            scheduler.train()
            score = test_agent(environments[i], scheduler)
            local_scores.append(score)

        global_dqn_scheduler = combine_agents(global_dqn_scheduler, dqn_schedulers, local_scores)
        dqn_schedulers = distribute_agents(global_dqn_scheduler, dqn_schedulers)

        score_single = test_agent(single_environment, single_dqn_scheduler)
        scores_single_agent.append(score_single)
        score_global = test_agent(global_environment, global_dqn_scheduler)
        scores_global_agent.append(score_global)

    pd.DataFrame({
        'Single Agent': scores_single_agent,
        'Global Agent': scores_global_agent
    }).to_csv('agent_scores_comparison.csv', index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(scores_single_agent, label='Single Agent', color='red')
    plt.plot(scores_global_agent, label='Aggregated Agent', color='green')
    plt.title('Performance Comparison')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.show()

if __name__ == '__main__':
    main()
