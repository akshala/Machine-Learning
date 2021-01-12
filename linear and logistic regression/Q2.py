from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np
np.seterr(all='ignore')

preProcessor = MyPreProcessor()
X, y = preProcessor.pre_process(2)

# Code for testing Q2 analysis values

logistic = MyLogisticRegression(alpha=0.01, n_iterations=1000, cost_fn='Stochastic')
X_train, y_train, X_val, y_val, X_test, y_test = logistic.train_val_test_split(X, y)
logistic.fit(X_train, y_train, X_val, y_val)

y_pred = logistic.predict(X_test)

accuracy = (y_pred == y_test).all(axis=1).mean()
print('testing accuracy', accuracy)