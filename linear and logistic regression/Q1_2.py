from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
import sys

preProcessor = MyPreProcessor()
X, y = preProcessor.pre_process(1)

# code for running Q1 Dataset2 analysis

linear = MyLinearRegression(alpha = 0.000001, n_iterations=2000, cost_fn='MAE')
linear.kFold(3, X, y)

linear = MyLinearRegression(alpha = 0.000001, n_iterations=2000, cost_fn='RMSE')
linear.kFold(3, X, y)