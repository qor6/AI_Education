#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from DataLoader import DataLoader


def simple_linear_regression_example():
    # Load the height, weight dataset
    height_x, weight_y = DataLoader.load_height_weight("./weight-height.csv")

    num_data = len(height_x)
    train_ratio = 0.8
    num_train = int(num_data * train_ratio)
    num_test = num_data - num_train

    # Split the data into training/testing sets
    height_x_train = height_x[:-num_test].reshape(-1, 1)
    height_x_test = height_x[num_train:].reshape(-1, 1)

    weight_y_train = weight_y[:-num_test].reshape(-1, 1)
    weight_y_test = weight_y[num_train:].reshape(-1, 1)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(height_x_train, weight_y_train)

    # Make predictions using the testing set
    weight_y_pred = regr.predict(height_x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(weight_y_test, weight_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(weight_y_test, weight_y_pred))

    # Plot outputs
    plt.scatter(height_x_test, weight_y_test,  color='black')
    plt.plot(height_x_test, weight_y_pred, color='blue', linewidth=3)

    plt.xticks(np.arange(130, 205, step=5))
    plt.yticks(np.arange(30, 105, step=5))

    plt.show()


if __name__ == '__main__':
    simple_linear_regression_example()
