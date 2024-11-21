import numpy as np

class EpsilonGreedyStruct:
    def __init__(self, featureDimension, lambda_, epsilon):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d) # 用来表示上下文特征的共变矩阵
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.b = np.zeros(self.d) # 用于累积的奖励
        self.AInv = np.linalg.inv(self.A) # A的逆矩阵
        self.UserTheta = np.zeros(self.d) # 根据上下文和点击数据更新得到的用户偏好参数向量
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector) #外积，结果是一个矩阵
        self.b += articlePicked_FeatureVector * click # r_t * x, r_t是奖励，x是特征向量
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        if self.epsilon is None:
            explore = np.random.binomial(1, (self.time+1)**(-1.0/3))
        else:
            explore = np.random.binomial(1, self.epsilon)
        if explore == 1:
            # print("EpsilonGreedy: explore")
            articlePicked = np.random.choice(pool_articles)
        else:
            # print("EpsilonGreedy: greedy")
            maxPTA = float('-inf')
            articlePicked = None

            for article in pool_articles:
                article_pta = np.dot(self.UserTheta, article.featureVector)
                # pick article with highest Prob
                if maxPTA < article_pta:
                    articlePicked = article
                    maxPTA = article_pta

        return articlePicked

class EpsilonGreedyLinearBandit:
    def __init__(self, dimension, lambda_, epsilon):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = EpsilonGreedyStruct(self.dimension, self.lambda_, self.epsilon)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


