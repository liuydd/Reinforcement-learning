import numpy as np

class UCBStruct:
    def __init__(self, num_arm, NoiseScale):
        self.d = num_arm
        self.UserArmMean = np.zeros(self.d) #记录每个臂的平均值（即每篇文章的点击率估计）
        self.UserArmTrials = np.zeros(self.d) #记录每个臂被选中的次数
        self.sigma = NoiseScale
        self.time = 0 #记录选择的总次数

    def updateParameters(self, articlePicked_id, click):
        self.UserArmMean[articlePicked_id] = (self.UserArmMean[articlePicked_id]*self.UserArmTrials[articlePicked_id] + click) / (self.UserArmTrials[articlePicked_id]+1)
        self.UserArmTrials[articlePicked_id] += 1
        self.time += 1

    def getTheta(self):
        return self.UserArmMean

    def decide(self, pool_articles):
        maxUCB = float('-inf')
        articlePicked = None

        for article in pool_articles:
            article_id = article.id
            
            # # 如果该臂从未被选中，强制探索
            if self.UserArmTrials[article_id] == 0:
                return article
            
            ucb = self.UserArmMean[article_id] + self.sigma * np.sqrt((2*np.log(self.time + 1)) / self.UserArmTrials[article_id])
            if ucb > maxUCB:
                articlePicked = article
                maxUCB = ucb

        return articlePicked
    
    
class UCBMultiArmedBandit:
    def __init__(self, num_arm, NoiseScale):
        self.users = {}
        self.num_arm = num_arm
        self.NoiseScale = NoiseScale
        self.CanEstimateUserPreference = False

    def decide(self, pool_articles, userID): #根据给定的用户 ID 和待选的文章池，做出一个选择
        if userID not in self.users:
            self.users[userID] = UCBStruct(self.num_arm, self.NoiseScale)

        return self.users[userID].decide(pool_articles)

    def updateParameters(self, articlePicked, click, userID): #更新用户的参数
        self.users[userID].updateParameters(articlePicked.id, click)

    def getTheta(self, userID): #返回指定用户的臂均值向量，表示对每个臂（文章）的偏好估计
        return self.users[userID].UserArmMean