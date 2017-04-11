import numpy as np
from DecisionTree import *


class RandomForest(object):

    def __init__(self, max_depth = 3, num_trees = 10, num_features = 10):
        '''

        :param max_depth: the maximum depth of a single decision tree
        :param num_trees: number of trees in the forest
        :param num_features: number of features use each time
        '''
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.num_features = num_features
        self.root = []
        self.split_rule = []

    def train(self, data, labels,feature_label):
        '''
        train the classifier by randomly choose subset rows and columns

        :param data: X
        :param labels: y
        :return: store each tree in the root list
        '''
        obs, vars = data.shape
        for tree_num in range(self.num_trees):
            print("TREE_num", tree_num)
            sample_obs = np.random.randint(0,obs,obs)
            sample_data = data[sample_obs,:]
            sample_label = labels[sample_obs]
            tree = DecisionTree(self.max_depth)
            tree.train(sample_data,sample_label,feature_label,random=True, num_features=self.num_features)
            self.split_rule.append((tree.sample_feature,tree_num))
            self.root += [tree]

    def predict(self,data,feat=None):
        '''
        predict test data

        :param data: validation or test data
        :return: mean of each decision tree prediction
        '''
        predict = []
        for t in self.root:
            pred = t.predict_prob(data,feat)
            predict.append(pred)
        return np.where(np.mean(np.array(predict),0)>0.5, 1, 0)
