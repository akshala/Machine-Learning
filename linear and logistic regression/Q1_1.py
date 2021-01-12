from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
import sys

preProcessor = MyPreProcessor()
X, y = preProcessor.pre_process(0)

# code for running Q1 dataset1 analysis

linear = MyLinearRegression(alpha=0.01, n_iterations=1000, cost_fn='MAE')
linear.kFold(3, X, y)

linear.loss_using_norm(X, y, 3)

linear = MyLinearRegression(alpha=0.01, n_iterations=1000, cost_fn='RMSE')
linear.kFold(3, X, y)
