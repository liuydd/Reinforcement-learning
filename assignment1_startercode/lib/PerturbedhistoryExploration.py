import numpy as np

class PHEStruct:
    def __init__(self, num_arm, alpha):
        self.d = num_arm
        self.UserArmMean = np.zeros(self.d) #记录每个臂的平均值（即每篇文章的点击率估计）
        self.UserArmTrials = np.zeros(self.d) #记录每个臂被选中的次数
        self.V = np.zeros(self.d)
        self.alpha = alpha
        self.time = 0 #记录选择的总次数

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1
        self.V[articlePicked_id] += click
        
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxPHE = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_id = article.id
            
            if self.UserArmTrials[article_id] > 0:
                s = self.UserArmTrials[article_id]
                pseudo_rewards = np.sum(np.random.binomial(int(self.alpha * s), 0.5))
                perturbed_mean = (self.V[article_id] + pseudo_rewards) / (s * self.alpha + s)
            else:
                perturbed_mean = float('inf')
                
            if perturbed_mean > maxPHE:
                articlePicked = article
                maxPHE = perturbed_mean

        return articlePicked
    
    
class PHEMultiArmedBandit:
    def __init__(self, num_arm, alpha):
        self.users = {}
        self.num_arm = num_arm
        self.alpha = alpha
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID): #根据给定的用户 ID 和待选的文章池，做出一个选择
        if userID not in self.users:
            self.users[userID] = PHEStruct(self.num_arm, self.alpha)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID): #更新用户的参数
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID): #返回指定用户的臂均值向量，表示对每个臂（文章）的偏好估计
        return self.users[userID].UserArmMean