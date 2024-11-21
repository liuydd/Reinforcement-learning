from MDP import build_mazeMDP, print_policy
import numpy as np

# task1
class DynamicProgramming:
	def __init__(self, MDP):
		self.R = MDP.R
		self.T = MDP.T
		self.discount = MDP.discount
		self.nStates = MDP.nStates
		self.nActions = MDP.nActions


	def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01): #TODO
		'''Value iteration procedure
		V <-- max_a R^a + gamma T^a V

		Inputs:
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		policy = np.zeros(self.nStates)
		# V = np.zeros(self.nStates)
		V = initialV.copy()
		iterId = 0
		epsilon = float("inf")
  
		while iterId < nIterations and epsilon > tolerance:
			V_old = V.copy()
			for s in range(self.nStates):
				policy[s] = np.argmax([self.R[a, s] + self.discount * np.dot(self.T[a, s, :], V) for a in range(self.nActions)])
			for s in range(self.nStates):
				V[s] = max([self.R[a, s] + self.discount * np.dot(self.T[a, s, :], V) for a in range(self.nActions)])
			epsilon = np.linalg.norm(V - V_old, np.inf)
			iterId += 1

		return [policy, V, iterId, epsilon]

	def policyIteration_v1(self, initialPolicy, nIterations=np.inf, tolerance=0.01): #TODO
		'''Policy iteration procedure: alternate between policy
		evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
		improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		nIterations -- limit on # of iterations: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		# policy = np.zeros(self.nStates)
		policy = initialPolicy.copy()
		V = np.zeros(self.nStates)
		iterId = 0
		policy_stable = False

		while iterId < nIterations and not policy_stable:
            # Policy Evaluation
			while True:
				V_prev = V.copy()
				A = np.zeros((self.nStates, self.nStates))
				b = np.zeros(self.nStates)
				for s in range(self.nStates):
					a = int(policy[s])  # Get the action from the policy
					A[s, s] = 1  # Coefficient for V[s]
					b[s] = self.R[a, s]  # Reward for taking action a in state s
					for s_prime in range(self.nStates):
						A[s, s_prime] -= self.discount * self.T[a, s, s_prime]  # Subtract discounted transition contributions
				V = np.linalg.solve(A, b)

				if np.max(np.abs(V - V_prev)) < tolerance:
					break
            
            # Policy Improvement
			policy_stable = True
			for s in range(self.nStates):
				old_action = policy[s]
				policy[s] = np.argmax([self.R[a, s] + self.discount * np.sum(self.T[a, s] * V) for a in range(self.nActions)])
				if old_action != policy[s]:
					policy_stable = False
            
			iterId += 1
		# while iterId < nIterations:
		# 	V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)
		# 	policy_new = self.extractPolicy(V)
		# 	iterId = iterId + 1

		# 	if np.array_equal(policy_new, policy):
		# 		break

		# 	policy = policy_new
		
		return [policy, V, iterId]


	def extractPolicy(self, V):
		'''Procedure to extract a policy from a value function
		pi <-- argmax_a R^a + gamma T^a V

		Inputs:
		V -- Value function: array of |S| entries

		Output:
		policy -- Policy: array of |S| entries'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		policy = np.zeros(self.nStates)
		for s in range(self.nStates):
			policy[s] = np.argmax([self.R[a, s] + self.discount * np.dot(self.T[a, s, :], V) for a in range(self.nActions)])
		return policy


	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi

		Input:
		policy -- Policy: array of |S| entries

		Ouput:
		V -- Value function: array of |S| entries'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		V = np.zeros(self.nStates)
		R_policy = np.array([self.R[int(policy[i])][i] for i in range(len(policy))])
		T_policy = np.array([self.T[int(policy[i])][i] for i in range(len(policy))])
		gamma_T_policy = self.discount * T_policy
		assert gamma_T_policy.shape[0] == gamma_T_policy.shape[1], "gamma_T_policy matrix should be square"
		V = np.matmul(np.linalg.inv(np.identity(len(policy)) - gamma_T_policy), R_policy)

		return V

	# nPolicyEvalIterations是可变的, 需要测试其在[1, 10]中iterId的数量
	def policyIteration_v2(self, initialPolicy, initialV, nPolicyEvalIterations=1, nIterations=np.inf, tolerance=0.01): #TODO
		'''Modified policy iteration procedure: alternate between
		partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
		and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nPolicyEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
		nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		# policy = np.zeros(self.nStates)
		policy = initialPolicy.copy()
		# V = np.zeros(self.nStates)
		V = initialV.copy()
		iterId = 0
		epsilon = float("inf")
		policy_stable = False
  
		while iterId < nIterations and epsilon > tolerance:
			# partial policy evaluation
			for _ in range(nPolicyEvalIterations):
				V_old = V.copy()
				for s in range(self.nStates):
					a = policy[s]
					V[s] = self.R[a,s] + self.discount * np.sum(self.T[a,s,:] * V_old)
				epsilon = np.max(np.abs(V - V_old))
			
			# policy improvement
			policy_stable = True
			for s in range(self.nStates):
				old_action = policy[s]
				policy[s] = np.argmax([self.R[a, s] + self.discount * np.sum(self.T[a, s, :] * V) for a in range(self.nActions)])
				if old_action != policy[s]:
					policy_stable = False
     
			iterId += 1
			if policy_stable:
				break
  
		# while iterId < nIterations and epsilon > tolerance:
		# 	iterId = iterId + 1
		# 	Vn, _ , _  = self.evaluatePolicy_IterativeUpdate(policy, V, nPolicyEvalIterations, tolerance)
		# 	all_possible_values = (self.R + (self.discount * np.matmul(self.T,Vn)))  # Get values for all possible state transition in this state
		# 	policy = np.argmax(all_possible_values, axis=0)  # Choose the best actions for each state, policy means keep

		# 	Vn_plus_1 = [all_possible_values[policy[i]][i] for i in range(len(policy))]
		# 	V_diff = (np.array(Vn_plus_1) - np.array(Vn))
		# 	V = Vn_plus_1
		# 	epsilon = np.linalg.norm(V_diff, np.inf)
   
			# iterId = iterId + 1
			# Vn, _ , _  = self.evaluatePolicy_IterativeUpdate(policy, V, nPolicyEvalIterations, tolerance)
			# for s in range(self.nStates):
			# 	policy[s] = np.argmax([self.R[a, s] + self.discount * np.sum(self.T[a, s, :] * Vn) for a in range(self.nActions)])
			# for s in range(self.nStates):
			# 	a = policy[s]
			# 	V[s] = self.R[a, s] + self.discount * np.sum(self.T[a, s, :] * Vn)
			# V_diff = (np.array(V) - np.array(Vn))
			# epsilon = np.linalg.norm(V_diff, np.inf)
  
		return [policy, V, iterId, epsilon]

	def evaluatePolicy_IterativeUpdate(self, policy, initialV, nIterations=np.inf, tolerance=0.01):
		'''Partial policy evaluation:
		Repeat V^pi <-- R^pi + gamma T^pi V^pi

		Inputs:
		policy -- Policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)

		Outputs:
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		V = np.zeros(self.nStates)
		iterId = 0
		epsilon = np.inf

		V = initialV

		while iterId < nIterations and epsilon > tolerance:
			iterId = iterId+1
			R_policy = np.array([self.R[policy[i]][i] for i in range(len(policy))])
			T_policy = np.array([self.T[policy[i]][i] for i in range(len(policy))])
			Vnew = R_policy + (self.discount * np.matmul(T_policy, V))
			epsilon = np.linalg.norm((Vnew-V), np.inf)
			V = Vnew

		return [V, iterId, epsilon]


if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), tolerance=0.01)
	print("Policy:", policy)
	print("Value function:", V)
	print("Number of iterations:", nIterations)
	print_policy(policy)
	# # Test policy iteration v1
	[policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, dtype=int))
	print("Policy:", policy)
	print("Value function:", V)
	print("Number of iterations:", nIterations)
	print_policy(policy)
	# # Test policy iteration v2
	[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates),nPolicyEvalIterations=4, tolerance=0.01)
	print("Policy:", policy)
	print("Value function:", V)
	print("Number of iterations:", nIterations)
	print_policy(policy)