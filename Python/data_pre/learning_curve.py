import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



train_sizes = [100, 200, 500, 1000, 2000, 5000, 6000]


def plotLearningCurve(estimator, title, X, y, cv=5, n_jobs=1,
                      train_sizes=train_sizes):
    """
    Generate learning curve of different models

    Parameters
    ------------------------------------------------------------------
    estimator : object type that implements the “fit” and “predict” methods

    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples) or (n_samples, n_features), optional

    train_sizes : array-like, shape (n_ticks,), dtype float or int
    cv : int, cross-validation generator or an iterable, optional

    n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
    title : string, plot title
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel('Training size')
    plt.ylabel('Score')

    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     color='blue', alpha=0.2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     color='red', alpha=0.2)

    plt.plot(train_sizes, train_mean, 'o-', color='blue',
             label='training error')
    plt.plot(train_sizes, test_mean, 'o-', color='r',
             label='validation error')

    plt.legend(loc='best')

    return plt
