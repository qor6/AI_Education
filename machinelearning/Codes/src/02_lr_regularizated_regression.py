#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Overfit Example
=========================================================
"""
print(__doc__)

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from DataLoader import DataLoader
import numpy as np


def make_6_order_feature_vecs(x):
    x1 = x
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    x6 = x5 * x

    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    x3 = x3.reshape(-1, 1)
    x4 = x4.reshape(-1, 1)
    x5 = x5.reshape(-1, 1)
    x6 = x6.reshape(-1, 1)
    features = np.concatenate((x6, x5, x4, x3, x2, x1), axis=1)
    return features


def print_coef(coef):
    coef_array = coef.flatten()
    for val in coef_array:
        print("%.4f" % val, end="\t")
    print("")


def linear_regression_overfit_example():

    # Load the wine features, wine quality dataset
    x, y = DataLoader.load_overfit_example()

    x_train = x.reshape(-1, 1)
    y_train = y.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    x_min = 0
    x_max = 10
    step = 0.2
    test_x = np.arange(x_min, x_max, step).reshape(-1, 1)
    y_pred = regr.predict(test_x)

    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('Training data')
    plt.show()

    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.plot(test_x, y_pred, color='blue', linewidth=3)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('One Feature Linear Regression')
    plt.show()

    features = make_6_order_feature_vecs(x)
    features = features.reshape(-1, 6)
    regr_multi = linear_model.LinearRegression()

    regr_multi.fit(features, y_train)

    # Make predictions using the testing set
    x_min = 0
    x_max = 10
    step = 0.2
    test_x = np.arange(x_min, x_max, step).reshape(-1, 1)
    test_features = make_6_order_feature_vecs(test_x)

    y_pred = regr_multi.predict(test_features)
    print_coef(regr_multi.coef_)
    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.plot(test_x, y_pred, color='blue', linewidth=3)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('Multi-Feature (6 order x ) with Regularization W = 0')
    plt.show()

    # L2 Regularization W = 0.1
    ridge = linear_model.Ridge(alpha=0.1)
    ridge.fit(features, y_train)
    y_pred = ridge.predict(test_features)
    print_coef(ridge.coef_)
    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.plot(test_x, y_pred, color='blue', linewidth=3)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('Multi-Feature (6 order x ) with L2 Reg. Lambda = 0.1')
    plt.show()

    # L2 Regularization W = 1
    ridge = linear_model.Ridge(alpha=1.0)
    ridge.fit(features, y_train)
    y_pred = ridge.predict(test_features)
    print_coef(ridge.coef_)
    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.plot(test_x, y_pred, color='blue', linewidth=3)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('Multi-Feature (6 order x ) with L2 Reg. Lambda = 1')
    plt.show()


    # L1 Regularization W = 0.1
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(features, y_train)
    y_pred = lasso.predict(test_features)
    print_coef(lasso.coef_)
    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.plot(test_x, y_pred, color='blue', linewidth=3)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('Multi-Feature (6 order x ) with L1 Reg. Lambda = 0.1')
    plt.show()


    # L1 Regularization W = 1
    lasso = linear_model.Lasso(alpha=1.0)
    lasso.fit(features, y_train)
    y_pred = lasso.predict(test_features)
    print_coef(lasso.coef_)
    # Plot outputs
    plt.scatter(x_train, y_train, color='black')
    plt.plot(test_x, y_pred, color='blue', linewidth=3)
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(6, 12, step=1))
    plt.ylim([5, 13])
    plt.title('Multi-Feature (6 order x ) with L1 Reg. Lambda = 1')
    plt.show()

if __name__ == '__main__':
    linear_regression_overfit_example()
