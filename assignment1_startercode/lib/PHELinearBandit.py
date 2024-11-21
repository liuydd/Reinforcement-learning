import numpy as np

# class PHEStruct: #alpha?
#     def __init__(self, featureDimension, lambda_, alpha):
#         self.d = featureDimension
#         self.alpha = alpha #随机扰动参数
#         self.A = (self.alpha + 1)*lambda_ * np.identity(n=self.d) # 用来表示上下文特征的共变矩阵
#         self.lambda_ = lambda_
#         self.b = np.zeros(self.d) # 用于累积的奖励
#         self.AInv = np.linalg.inv(self.A) # A的逆矩阵
#         self.UserTheta = np.zeros(self.d) # 根据上下文和点击数据更新得到的用户偏好参数向量
#         self.time = 0
#         self.armfeatureVector = {}
#         self.armTrials = {}
#         self.armReward = {}
#         self.f_noise =np.zeros(self.d)
        

#     def updateParameters(self, articlePicked, click):
#         if articlePicked.id not in self.armTrials:
#             self.armTrials[articlePicked.id] = 0
#             self.armfeatureVector[articlePicked.id] = articlePicked.featureVector
#             self.armReward[articlePicked.id] = 0
#         self.armTrials[articlePicked.id] += 1
#         self.armReward[articlePicked.id] += click
  
#         self.A += (self.alpha + 1)*np.outer(articlePicked.featureVector, articlePicked.featureVector) #外积，结果是一个矩阵
#         self.AInv = np.linalg.inv(self.A)
#         perturbed_f = np.zeros(self.d)
#         for armID, reward in self.armReward.items():
#             pseudo_rewards = np.random.binomial(self.alpha * self.armTrials[armID], 0.5)
#             perturbed_f += self.armfeatureVector[armID] * (reward + pseudo_rewards)
#         self.f_noise += articlePicked.featureVector * click
        
#         self.UserTheta = np.dot(self.AInv, perturbed_f)
#         self.time += 1

#     def getTheta(self):
#         return np.dot(self.AInv, self.f_noise)

#     def getA(self):
#         return self.A
    

#     def decide(self, pool_articles):
#         maxPTA = float('-inf')
#         articlePicked = None

#         for article in pool_articles:
#             if article.id not in self.armTrials:
#                 return article
            
#             article_pta = np.dot(self.UserTheta, article.featureVector)
            
#             if maxPTA < article_pta:
#                 articlePicked = article
#                 maxPTA = article_pta

#         return articlePicked


# class PHELinearBandit:
#     def __init__(self, dimension, lambda_, alpha):
#         self.users = {}
#         self.dimension = dimension
#         self.lambda_ = lambda_
#         self.CanEstimateUserPreference = True
#         self.alpha = alpha

#     def decide(self, pool_articles, userID):
#         if userID not in self.users:
#             self.users[userID] = PHEStruct(self.dimension, self.lambda_, self.alpha)

#         return self.users[userID].decide(pool_articles)

#     def updateParameters(self, articlePicked, click, userID):
#         self.users[userID].updateParameters(articlePicked, click)

#     def getTheta(self, userID):
#         return self.users[userID].UserTheta

class LinPHEStruct:
	def __init__(self, featureDimension, lambda_, a):
		self.d = featureDimension
		self.lambda_ = lambda_
		self.time = 0

		self.armFeatureVecs = {}
		self.armTrials = {}
		self.armCumReward = {}
		self.a = a
		# if a != -1:
		# 	self.a = a
		# else:
		# 	# see Perturbed-History Exploration in Stochastic Linear Bandits Table 1
		# 	c1 = 0.5 * np.sqrt(
		# 		self.d * np.log(testing_iterations + testing_iterations**2 / (self.d * self.lambda_))) + np.sqrt(self.lambda_)
		# 	self.a = math.ceil(16 * c1**2)
		print("self.a {}".format(self.a))
		self.f_noiseless = np.zeros(self.d)
		self.B = np.zeros((self.d, self.d))
		self.UserTheta = np.zeros(self.d)

		self.G_0 = lambda_ * (self.a + 1) * np.identity(self.d)

	def updateParameters(self, article_picked, click):
		self.time += 1
		if article_picked.id not in self.armTrials:
			self.armTrials[article_picked.id] = 0
			self.armFeatureVecs[article_picked.id] = article_picked.featureVector
			self.armCumReward[article_picked.id] = 0
		self.armTrials[article_picked.id] += 1
		self.armCumReward[article_picked.id] += click

		self.B += np.outer(article_picked.featureVector, article_picked.featureVector)
		G = (self.a + 1) * self.B + self.G_0

		perturbed_f = np.zeros(self.d)
		for armID, armCumReward in self.armCumReward.items():
			perturbed_f += self.armFeatureVecs[armID] * (armCumReward + np.random.binomial(self.a*self.armTrials[armID], 0.5))

		self.f_noiseless += article_picked.featureVector * click
		self.UserTheta = np.dot(np.linalg.inv(G), perturbed_f)

	def getProb(self, article_featureVector):
		return np.dot(self.UserTheta, article_featureVector)

	def getTheta(self):
		return np.dot(np.linalg.inv(self.B+self.lambda_*np.identity(self.d)), self.f_noiseless)

class PHELinearBandit:
	def __init__(self, dimension, lambda_, alpha=1):
		self.dimension = dimension
		self.alpha = alpha
		self.lambda_ = lambda_

		self.users = {}

		self.CanEstimateUserPreference = True

	def decide(self, pool_articles, userID):
		if userID not in self.users:
			self.users[userID] = LinPHEStruct(self.dimension, self.lambda_, self.alpha)

		maxPTA = float('-inf')
		articlePicked = None


		for x in pool_articles:
			if x.id not in self.users[userID].armTrials:
				return x
			x_pta = self.users[userID].getProb(x.featureVector)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta
		return articlePicked

	def updateParameters(self, article_picked, click, userID):
		self.users[userID].updateParameters(article_picked, click)

	def getTheta(self, userID):
		return self.users[userID].getTheta()