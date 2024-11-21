from MDP import build_mazeMDP, print_policy
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

	def sampleRewardAndNextState(self,state,action):
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

		reward = self.sampleReward(self.mdp.R[action,state])
		cumProb = np.cumsum(self.mdp.T[action,state,:])
		nextState = np.where(cumProb >= np.random.rand(1))[0][0]
		return [reward,nextState]

	def OffPolicyTD(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy TD (Q-learning) algorithm
		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random
		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		policy = np.zeros(self.mdp.nStates,int)
		n = np.zeros((self.mdp.nActions, self.mdp.nStates))
		cumulative_rewards = np.zeros(nEpisodes)

		for episode in range(nEpisodes):
			state = np.random.randint(self.mdp.nStates)  # Start from a random state
			for step in range(100):
			# done = False
			# step = 0
			# while not done:
				# Choose an action using epsilon-greedy policy
				if np.random.rand() < epsilon:
					action = np.random.randint(self.mdp.nActions)
				else:
					action = np.argmax(Q[:, state])
                
				reward, nextState = self.sampleRewardAndNextState(state, action)
				cumulative_rewards[episode] += reward * self.mdp.discount ** step
				n[action][state] += 1
				alpha = 1.0 / n[action][state]
				Q[action, state] += alpha * (reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, state])

				state = nextState
				# step += 1
				if np.random.rand() < 0.1:  # Condition to end episode (e.g., reaching terminal state)
					done = True

        # Derive policy from Q
		policy = np.argmax(Q, axis=0)

		return [Q,policy, cumulative_rewards]

	def OffPolicyMC(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy MC algorithm with epsilon-soft behavior policy
		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random
		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		# Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		# policy = np.zeros(self.mdp.nStates,int)

		# return [Q,policy]
		Q = np.zeros([self.mdp.nActions, self.mdp.nStates])
		C = np.zeros([self.mdp.nActions, self.mdp.nStates])
		N = np.zeros([self.mdp.nActions, self.mdp.nStates])
		policy = np.zeros(self.mdp.nStates, int)
		cumulative_rewards = np.zeros(nEpisodes)

		for episode in range(nEpisodes):
			state = np.random.randint(self.mdp.nStates)  # Start from a random state
			# done = False
			episode_data = []
			for step in range(100):
			# step = 0
			# while not done:
				# Choose an action using epsilon-soft policy
				if np.random.rand() < epsilon:
					action = np.random.randint(self.mdp.nActions)
				else:
					action = np.argmax(Q[:, state])

				# Sample reward and next state
				reward, nextState = self.sampleRewardAndNextState(state, action)
				episode_data.append((state, action, reward))
				cumulative_rewards[episode] += reward * self.mdp.discount ** step
				state = nextState
				# step += 1
				# if np.random.rand() < 0.1:  # Condition to end episode
				# 	done = True
            
			G = 0
			W = 1 
			for state, action, reward in reversed(episode_data):
				G = reward + G * self.mdp.discount
				C[action, state] += W
				N[action, state] += 1
				Q[action, state] += (W / N[action, state]) * (G - Q[action, state])
				if action != np.argmax(Q[:, state]):
					break 
				W *= (1 - epsilon) / (1 - (1 - epsilon) / self.mdp.nActions)

		policy = np.argmax(Q, axis=0)

		return [Q, policy, cumulative_rewards]

	def drawMC(self, nEpisodes, epsilon):
		Run = 100
		avg_cumulative_rewards = np.zeros(nEpisodes)
		for i in range(Run):
			[Q, policy, cumulative_rewards] = self.OffPolicyMC(nEpisodes=nEpisodes, epsilon=epsilon)
			avg_cumulative_rewards += cumulative_rewards
		avg_cumulative_rewards /= Run
		# plt.plot(avg_cumulative_rewards)
		# plt.show()
		plt.title("off-policy MC control")
		plt.xlabel("Episode #")
		plt.ylabel("Average Cumulative Discounted Rewards")
		xAxis = range(1, (len(avg_cumulative_rewards)+1))
		yAxis = avg_cumulative_rewards
		pltLabel = "Epsilon : " + str(epsilon)
		plt.plot(xAxis, yAxis, label= pltLabel)
		plt.show()
  
	def drwaTD(self, nEpisodes):
		Run = 100
		avg_cumulative_rewards = np.zeros(nEpisodes)
		fig = plt.figure()
		plt.title("off-policy TD control")
		plt.xlabel("Episode #")
		plt.ylabel("Average Cumulative Discounted Rewards")
  
		for i in range(Run):
			[Q, policy, cumulative_rewards] = self.OffPolicyTD(nEpisodes=nEpisodes, epsilon=0.05)
			avg_cumulative_rewards += cumulative_rewards
		avg_cumulative_rewards /= Run
		xAxis = range(1, (len(avg_cumulative_rewards)+1))
		yAxis = avg_cumulative_rewards
		pltLabel = "Epsilon : " + str(0.05)
		plt.plot(xAxis, yAxis, label= pltLabel)
  
		for i in range(Run):
			[Q, policy, cumulative_rewards] = self.OffPolicyTD(nEpisodes=nEpisodes, epsilon=0.1)
			avg_cumulative_rewards += cumulative_rewards
		avg_cumulative_rewards /= Run
		xAxis = range(1, (len(avg_cumulative_rewards)+1))
		yAxis = avg_cumulative_rewards
		pltLabel = "Epsilon : " + str(0.1)
		plt.plot(xAxis, yAxis, label= pltLabel)
  
		for i in range(Run):
			[Q, policy, cumulative_rewards] = self.OffPolicyTD(nEpisodes=nEpisodes, epsilon=0.3)
			avg_cumulative_rewards += cumulative_rewards
		avg_cumulative_rewards /= Run
		xAxis = range(1, (len(avg_cumulative_rewards)+1))
		yAxis = avg_cumulative_rewards
		pltLabel = "Epsilon : " + str(0.3)
		plt.plot(xAxis, yAxis, label= pltLabel)
  
		for i in range(Run):
			[Q, policy, cumulative_rewards] = self.OffPolicyTD(nEpisodes=nEpisodes, epsilon=0.5)
			avg_cumulative_rewards += cumulative_rewards
		avg_cumulative_rewards /= Run
		xAxis = range(1, (len(avg_cumulative_rewards)+1))
		yAxis = avg_cumulative_rewards
		pltLabel = "Epsilon : " + str(0.5)
		plt.plot(xAxis, yAxis, label= pltLabel)
  
		plt.legend(loc='best')
		# plt.savefig('TD.png')
		plt.show()
		plt.close()

if __name__ == '__main__':
	mdp = build_mazeMDP()
	rl = ReinforcementLearning(mdp, np.random.normal)

	# # Test Q-learning
	# [Q, policy, cumulative_rewards] = rl.OffPolicyTD(nEpisodes=500, epsilon=0.1)
	# print_policy(policy)

	# # Test Off-Policy MC
	# [Q, policy, cumulative_rewards] = rl.OffPolicyMC(nEpisodes=500, epsilon=0.1)
	# print_policy(policy)
	rl.drawMC(nEpisodes=500, epsilon=0.1)
	# rl.drwaTD(nEpisodes=500)
