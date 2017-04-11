import numpy as np
import statistics as stat
from Node import *
from collections import Counter


class DecisionTree(object):

    def __init__(self, max_depth =3):
        self.nodes = 1
        self.max_depth = max_depth
        self.sample_feature = []


    def entropy(self, child):
        '''
        calculate the entropy of each node

        :param child: {'left' : freq; 'right' : freq}
        :return: a scalar: entropy
        '''
        freq_l, freq_r = child.values()
        if freq_l == 0:
            return -np.log2(1)
        elif freq_r == 0:
            return -np.log2(1)
        else:
            pl = freq_l / (freq_l + freq_r)
            pr = 1 - pl
            return - pl * np.log2(pl) - pr * np.log2(pr)


    def impurity(self, left_label_hist, right_label_hist):
        '''
        calculate the weighed avg entropy after each split

        :param left_label_hist: {'left': freq, 'right': freq}
        :param right_label_hist: {'left': freq,'right': freq}
        :return: a scalar: weighted avg entropy after split
        '''
        left_num = left_label_hist['left'] + left_label_hist['right']
        right_num = right_label_hist['left'] + right_label_hist['right']
        total_num = left_num + right_num
        H_after = (left_num / total_num) * self.entropy(left_label_hist) + (right_num / total_num) * self.entropy(right_label_hist)
        return H_after


    def feature_split(self, labels, feature_name, feature,feature_label):
        '''
        find best split feature by find minimum impurity

        :param labels: y
        :param feature_name: split feature
        :param feature: feature value
        :return: a list of tuple with split feature and impurity
        '''
        split = []
        feature_impurity = []
        for i in np.unique(feature):
            left_child = labels[feature <= i]
            left_ones = np.count_nonzero(left_child)
            left_zeros = left_child.shape[0] - left_ones

            right_child = labels[feature > i]
            right_ones = np.count_nonzero(right_child)
            right_zeros = right_child.shape[0] - right_ones

            left_label_hist = {'left': left_ones, 'right': left_zeros}
            right_label_hist = {'left': right_ones, 'right': right_zeros}

            H_child = self.impurity(left_label_hist, right_label_hist)
            split_name = "{} <= {} ".format(feature_label[int(feature_name)], i)
            split.append(split_name)
            feature_impurity.append((split_name, H_child))
        return feature_impurity

    def segmenter(self, data, labels,feature_label, random ,num_features):
        '''
        find best value to split

        :param data: X
        :param labels: y
        :return: a tuple of split feature and split value
        '''

        information_gain = []
        if not random :
            for feature_name, feature in enumerate(data.T):
                score = self.feature_split(labels, feature_name, feature,feature_label)
                information_gain.extend(score)
            info_gain = min(information_gain, key=lambda x: x[1])
            feat, val = info_gain
            return (feat.split()[0], float(feat.split()[2]))
        else:
            sample_vars = np.random.choice(data.shape[1], num_features, replace=False)
            sample_data = data[:,sample_vars]
            for feature_name, feature in enumerate(sample_data.T):
                score = self.feature_split(labels, feature_name, feature,feature_label)
                information_gain.extend(score)
            info_gain = min(information_gain, key=lambda x: x[1])
            self.sample_feature += info_gain
            feat, val = info_gain

            index = sample_vars[feature_label.index(feat.split()[0])]
            return (feature_label[index], float(feat.split()[2]),sample_vars)


    def build_tree(self, data, labels,feature_label,random, num_features,depth = 1 ):
        '''
        grow tree by recursion

        :param data: X
        :param labels: y
        :param depth: the depth of tree
        :return: Node or leaf
        '''
        ones = np.count_nonzero(labels)
        zeros = len(labels) - ones

        if ones/len(labels) >= 0.99 or zeros/len(labels) >= 0.99:
            if zeros/len(labels) >= 0.99:
                return  Node(labels = 0, node_type='leaf')
            if ones/len(labels) >= 0.99:
                return Node(labels = 1, node_type='leaf')
        elif self.max_depth == depth:
            return Node(labels = Counter(labels).most_common(1)[0][0], node_type='leaf')
        elif len(labels) <= 10:
            return Node(labels=Counter(labels).most_common(1)[0][0], node_type='leaf')
        else:
            if not random:
                root = Node(labels, node_type='node')
                feat, val = self.segmenter(data, labels,feature_label,random, num_features)
                index = data[..., list(feature_label).index(feat)] <= val
                lX, ly = data[index], labels[index]
                rX, ry = data[~index], labels[~index]
                if len(ly) == 0 or len(ry) == 0:
                    return Node(labels =Counter(labels).most_common(1)[0][0], node_type='leaf')
                root.split_rule = (list(feature_label).index(feat),val)
                root.left = self.build_tree(lX, ly,feature_label,random, num_features, depth+1)
                root.right = self.build_tree(rX, ry,feature_label,random, num_features,depth+1)
            else:
                root = Node(labels, node_type='node')
                feat, val,sample_var = self.segmenter(data, labels, feature_label,random,num_features)
                index = data[..., feature_label.index(feat)] <= val
                lX, ly = data[index], labels[index]
                rX, ry = data[~index], labels[~index]
                if len(ly) == 0 or len(ry) == 0:
                    return Node(labels=Counter(labels).most_common(1)[0][0], node_type='leaf')
                root.split_rule = (list(feature_label).index(feat), val)
                root.left = self.build_tree(lX, ly, feature_label,random, num_features, depth + 1)
                root.right = self.build_tree(rX, ry, feature_label,random, num_features, depth + 1)

            return root



    def train(self, data, labels,feature_label=None,random=None, num_features=None):
        '''

        :param data: X
        :param labels: y
        :return: store a tree
        '''
        self.root = self.build_tree(data, labels,feature_label,random, num_features)

    def predict_prob(self, data,feature_label):
        pred_prob = np.array([self.root.traversing(obs,feature_label) for obs in data])
        return pred_prob

    def predict(self, data,feature_label):
        return np.where(self.predict_prob(data,feature_label) > 0.5, 1, 0)

