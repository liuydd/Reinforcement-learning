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

    def policy(self, theta, state):
        """Softmax policy."""
        logits = theta[:, state]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def reinforce(self, theta, alpha=0.001, nEpisodes=3000):
        n_actions, n_states = self.mdp.R.shape
        if theta is None:
            theta = np.random.rand(n_states, n_actions)

        cum_rewards = np.zeros(nEpisodes)

        for episode in range(nEpisodes):
            # state = np.random.choice(range(n_states))
            state = 0
            trajectory = []
            done = False

            # Generate episode
            while not done:
                probs = self.policy(theta, state)
                action = np.random.choice(range(n_actions), p=probs)
                reward, next_state = self.sampleRewardAndNextState(state, action)
                trajectory.append((state, action, reward))
                cum_rewards[episode] += reward
                state = next_state

                if state == self.mdp.nStates - 1:
                    done = True

            # Policy gradient update
            for t, (s, a, r) in enumerate(trajectory):
                G_t = sum([(self.mdp.discount ** k) * reward for k, (_, _, reward) in enumerate(trajectory[t:])])
                probs = self.policy(theta, s)
                grad_log_pi = -probs
                grad_log_pi[a] += 1
                theta[:, s] += alpha * (self.mdp.discount ** t) * G_t * grad_log_pi

        return cum_rewards, theta

    def actorCritic(self, theta, alpha=0.001, beta=0.01, nEpisodes=3000):
        n_actions, n_states = self.mdp.R.shape
        if theta is None:
            theta = np.random.rand(n_states, n_actions)
        w = np.zeros(n_states)  # Value function parameters

        cum_rewards = np.zeros(nEpisodes)

        for episode in range(nEpisodes):
            # state = np.random.choice(range(n_states))
            state = 0
            I = 1
            done = False

            while not done:
                probs = self.policy(theta, state)
                action = np.random.choice(range(n_actions), p=probs)
                reward, next_state = self.sampleRewardAndNextState(state, action)

                # TD error
                td_error = reward + self.mdp.discount * w[next_state] - w[state]

                # Critic update
                w[state] += beta * td_error

                # Actor update
                grad_log_pi = -probs
                grad_log_pi[action] += 1
                theta[:, state] += alpha * I * td_error * grad_log_pi

                cum_rewards[episode] += reward
                I = self.mdp.discount * I
                state = next_state

                if state == self.mdp.nStates - 1:
                    done = True
                      

        return cum_rewards, theta


# mdp = build_mazeMDP(b=0.1)
# rl = ReinforcementLearning(mdp, np.random.normal)

# n_episode = 3000
# n_trials = 10
# n_states = mdp.R.shape[1]
# n_actions = mdp.R.shape[0]
# init_theta = np.zeros((n_actions, n_states))

# # # Test PG
# out = np.zeros([n_trials, n_episode])
# for i in range(n_trials):
# 	cum_rewards, theta = rl.reinforce(theta=init_theta, alpha=0.003, nEpisodes=n_episode)
# 	out[i, :] = np.array(cum_rewards)
# plt.plot(out.mean(axis=0), label='Reinforce')

# # # Test AC
# out = np.zeros([n_trials, n_episode])
# for i in range(n_trials):
# 	# [cum_rewards, policy_ac, v] = rl.actorCritic()
# 	cum_rewards, theta = rl.actorCritic(theta=init_theta, alpha=0.001, beta=0.01, nEpisodes=n_episode)
# 	out[i] = cum_rewards
# plt.plot(out.mean(axis=0), label='ActorCritic')
# plt.legend()

# plt.show()


b_values = [0.0, 0.1, 0.2, 0.3]
n_episode = 3000
n_trials = 10

plt.figure(figsize=(10, 6))

for b in b_values:
    mdp = build_mazeMDP(b=b)
    rl = ReinforcementLearning(mdp, np.random.normal)
    
    n_states = mdp.R.shape[1]
    n_actions = mdp.R.shape[0]
    init_theta = np.zeros((n_actions, n_states))
    
    # 存储结果
    out = np.zeros([n_trials, n_episode])
    for i in range(n_trials):
        # cum_rewards, theta = rl.reinforce(theta=init_theta, alpha=0.001, nEpisodes=n_episode)
        cum_rewards, theta = rl.actorCritic(theta=init_theta, alpha=0.001, beta=0.01, nEpisodes=n_episode)
        out[i, :] = np.array(cum_rewards)
    
    # 计算平均累计奖励
    avg_cum_rewards = out.mean(axis=0)
    
    # 绘制曲线
    plt.plot(avg_cum_rewards, label=f'b = {b}')

# 添加图例、标题和轴标签
plt.legend()
plt.title('Actor-Critic')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Rewards')
plt.grid(True)
plt.show()