
import os
from concurrent.futures import ProcessPoolExecutor
from Multi_Armed_Bandit.bandit import generate_multi_armed_bandits, Bandit
from Multi_Armed_Bandit.policy import EGreedy, DecreasingEGreedy, UCB1, ThompsonSampling
from constant import PROJECT_HOME


package_home = os.path.join(PROJECT_HOME, "Multi_Armed_Bandit")
output_path = os.path.join(package_home, "output")
os.makedirs(output_path, exist_ok=True)

N = 10
T = 100000
SEED = 2019
policies = {
    "e_greedy": EGreedy(epsilon=0.05, seed=SEED),
    "decreasing_e_greedy": DecreasingEGreedy(seed=SEED),
    "Thompson_sampling": ThompsonSampling(10, 20),
    "ucb1": UCB1(),
}


def evaluation(policy_name: str):
    bandits = generate_multi_armed_bandits(N, SEED)
    policy = policies[policy_name]
    choices_log = []
    for _ in range(T):
        score = policy.step(bandits)
        choices_log.append(score)
    avg_score = sum(choices_log) / len(choices_log)
    with open(os.path.join(output_path, f"{policy_name}.txt"), "w") as f:
        for score in choices_log:
            f.write(f"{score}\n")
    print(f"{policy_name}: {avg_score:.5f}")


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=len(policies)) as pool:  # to guarantee the env equality for each policy
        pool.map(evaluation, policies.keys(),)
