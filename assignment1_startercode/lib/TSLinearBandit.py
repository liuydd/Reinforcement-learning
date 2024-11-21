import numpy as np

class TSStruct:
    def __init__(self, featureDimension, lambda_, sigma):
        self.d = featureDimension
        self.A = lambda_ * np.identity(n=self.d) # 用来表示上下文特征的共变矩阵
        self.lambda_ = lambda_
        self.b = np.zeros(self.d) # 用于累积的奖励
        self.AInv = np.linalg.inv(self.A) # A的逆矩阵
        self.UserTheta = np.zeros(self.d) # 根据上下文和点击数据更新得到的用户偏好参数向量
        self.sigma2 = sigma * sigma
        self.time = 0

    def updateParameters(self, articlePicked_FeatureVector, click):
        self.A += (1/self.sigma2) * np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector) #外积，结果是一个矩阵
        self.b += (1/self.sigma2) * articlePicked_FeatureVector * click # r_t * x, r_t是奖励，x是特征向量
        self.AInv = np.linalg.inv(self.A)
        self.UserTheta = np.dot(self.AInv, self.b)
        self.time += 1

    def getTheta(self):
        return self.UserTheta

    def getA(self):
        return self.A

    def decide(self, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        sample_theta = np.random.multivariate_normal(self.UserTheta, self.lambda_ * self.AInv)
        
        for article in pool_articles:
            article_pta = np.dot(sample_theta, article.featureVector)
            
            if maxPTA < article_pta:
                articlePicked = article
                maxPTA = article_pta

        return articlePicked

class TSLinearBandit:
    def __init__(self, dimension, lambda_, sigma):
        self.users = {}
        self.dimension = dimension
        self.lambda_ = lambda_
        self.sigma = sigma
        self.CanEstimateUserPreference = True

    def decide(self, pool_articles, userID):
        if userID not in self.users:
            self.users[userID] = TSStruct(self.dimension, self.lambda_, self.sigma)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID):
        self.users[userID].updateParameters(articlePicked.featureVector[:self.dimension], click)

    def getTheta(self, userID):
        return self.users[userID].UserTheta


