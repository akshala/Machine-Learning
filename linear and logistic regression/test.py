from scratch import MyLinearRegression, MyLogisticRegression, MyPreProcessor
import numpy as np

preprocessor = MyPreProcessor()

print('Linear Regression')

X, y = preprocessor.pre_process(0)

# Create your k-fold splits or train-val-test splits as required

linear = MyLinearRegression()
linear.kFold(3, X, y)

print('Logistic Regression')

X, y = preprocessor.pre_process(2)

# Create your k-fold splits or train-val-test splits as required

logistic = MyLogisticRegression()
X_train, y_train, X_val, y_val, X_test, y_test = logistic.train_val_test_split(X, y)
logistic.fit(X_train, y_train, X_val, y_val)

ypred = logistic.predict(X_test)

print('Predicted Values:', ypred)
print('True Values:', y_test)