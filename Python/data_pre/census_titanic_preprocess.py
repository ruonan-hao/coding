from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
import csv
import scipy.io
# from scipy import


def vectorizer(file, y_label):
    data = pd.read_csv(file)
    catgory = []
    numerical = []
    mode = {}
    for i in data.columns:
        if data[i].dtype == 'int64' or data[i].dtype == 'float64':
            numerical.append(i)
        else:
            catgory.append(i)
        mode[i] = (data[i].value_counts().index[0])
    xx = []
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in mode.keys():
                if key == 'ticket' or key == 'cabin':
                    row[key] = 0
                if row[key] == '?' or row[key] == '':
                    row[key] = mode[key]
                if key in numerical:
                    row[key] = float(row[key])
            xx.append(row)
    v = DictVectorizer(sparse=False)
    result = v.fit_transform(xx)
    feature = v.get_feature_names()
    index = feature.index(y_label)
    result_remove_label = np.hstack([result[..., range(index)], result[..., range(index + 1, result.shape[1])]])
    feature.remove(y_label)
    return {'feature_name': feature, 'data': result_remove_label,
            'label': result[:, index]}


def test_vectorizer(file):
    data = pd.read_csv(file)
    catgory = []
    numerical = []
    mode = {}
    for i in data.columns:
        if data[i].dtype == 'int64' or data[i].dtype == 'float64':
            numerical.append(i)
        else:
            catgory.append(i)
        mode[i] = (data[i].value_counts().index[0])
    xx = []
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in mode.keys():
                if key == 'ticket' or key == 'cabin':
                    row[key] = 0
                if row[key] == '?' or row[key] == '':
                    row[key] = mode[key]

                if key in numerical:
                    row[key] = float(row[key])
            xx.append(row)
    v = DictVectorizer(sparse=False)
    result = v.fit_transform(xx)
    feature = v.get_feature_names()
    return {'feature_name': feature, 'data': result}

census = vectorizer('hw5_census_dist/train_data.csv','label')
scipy.io.savemat('census_.mat',census)

titanic = vectorizer('hw5_titanic_dist/titanic_training.csv','survived')
titanic_test = test_vectorizer('hw5_titanic_dist/titanic_testing_data.csv')
scipy.io.savemat('titanic_.mat',titanic)
scipy.io.savemat('titanic_test_.mat',titanic_test)