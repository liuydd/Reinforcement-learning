from MDP import build_mazeMDP
import numpy as np
import matplotlib.pyplot as plt

class ReinforcementLearning:
    def __init__(self, mdp, sampleReward):
        """
		Constructor for the RL class

		:param mdp: Markov decision process (T, R, discount)
		:param sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian). This function takes one argument:
		the mean of the distribution and returns a sample from the distribution.
		"""

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
		reward ~ Pr(r)
		nextState ~ Pr(s'|s,a)

		Inputs:
		state -- current state
		action -- action to be executed

		Outputs:
		reward -- sampled reward
		nextState -- sampled next state
		'''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def reinforce(self, theta=None):

        # initialize policy parameterization

        cum_rewards = []

        return cum_rewards, theta

    def actorCritic(self, theta=None):

        # initialize policy parameterization

        cum_rewards = []

        return cum_rewards, theta


mdp = build_mazeMDP(b=0.1)
rl = ReinforcementLearning(mdp, np.random.normal)

n_episode = 3000
n_trials = 10

# Test PG
out = np.zeros([n_trials, n_episode])

for i in range(n_trials):
	cum_rewards, theta = rl.reinforce()
	out[i, :] = np.array(cum_rewards)
plt.plot(out.mean(axis=0), label='Reinforce')

# Test AC
out = np.zeros([n_trials, n_episode])
for i in range(n_trials):
	[cum_rewards, policy_ac, v] = rl.actorCritic()
	out[i] = cum_rewards
plt.plot(out.mean(axis=0), label='ActorCritic')
plt.legend()

plt.show()

