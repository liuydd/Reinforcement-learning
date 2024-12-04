from PG import ReinforcementLearning
from RL import ReinforcementLearning_MP2
from MDP import build_mazeMDP
import numpy as np
import matplotlib.pyplot as plt

mdp = build_mazeMDP(b=0.0)
rl = ReinforcementLearning(mdp, np.random.normal)
rl_mp2 = ReinforcementLearning_MP2(mdp, np.random.normal)

n_episode = 3000
n_trials = 10

# Test PG
out = np.zeros([n_trials, n_episode])
n_states = mdp.R.shape[1]
n_actions = mdp.R.shape[0]
theta = np.zeros((n_actions, n_states))

plt.figure(figsize=(10, 6))
for i in range(n_trials):
	cum_rewards, theta = rl.reinforce(theta, alpha=0.001, nEpisodes=n_episode)
	out[i, :] = np.array(cum_rewards)
plt.plot(out.mean(axis=0), label='Reinforce')

for i in range(n_trials):
	cum_rewards, theta = rl.actorCritic(theta=theta, alpha=0.001, beta=0.01, nEpisodes=n_episode)
	out[i] = cum_rewards
plt.plot(out.mean(axis=0), label='ActorCritic')

for i in range(n_trials):
	[Q, policy, cumulative_rewards] = rl_mp2.OffPolicyMC(nEpisodes=n_episode, epsilon=0.1)
	out[i, :] = np.array(cumulative_rewards)
plt.plot(out.mean(axis=0), label='OffPolicyMC')

for i in range(n_trials):
	[Q, policy, cumulative_rewards] = rl_mp2.OffPolicyTD(nEpisodes=n_episode, epsilon=0.1)
	out[i, :] = np.array(cumulative_rewards)
plt.plot(out.mean(axis=0), label='OffPolicyTD')

plt.legend()

plt.show()