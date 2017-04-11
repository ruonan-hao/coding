import numpy as np

class Node(object):
    def __init__(self,labels, split_rule=(None,None), left = -2, right = -2, node_type = None):
        '''

        :param split_rule: a tuple contains split_feature and threshold
        :param labels: y
        :param left: left child of a tree
        :param right: right child of a tree
        '''
        self.labels = labels
        self.node_type = node_type
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.split_name = []

    def __repr__(self):
        return self.node_type

    def prob(self):
        '''

        :return: return probability if it's a leaf
        '''
        total = len(self.labels)
        ones = np.count_nonzero(self.labels)
        return ones/ total

    def traversing(self, obs ,feature_label):
        '''

        :param obs: each predict data
        :return: probability
        '''
        if obs[int(self.split_rule[0])] <= self.split_rule[1]:
            # print(obs)
            split_name = "{} <= {} ".format(feature_label[int(self.split_rule[0])],self.split_rule[1])
            print(split_name)
            self.split_name.append(split_name)
            if self.left.node_type == 'leaf':
                return self.prob()
            return self.left.traversing(obs,feature_label)
        else:
            split_name = "{} > {} ".format(feature_label[int(self.split_rule[0])], self.split_rule[1])
            print(split_name)
            self.split_name.append(split_name)
            if self.right.node_type == 'leaf':
                return self.prob()
            return self.right.traversing(obs,feature_label)
