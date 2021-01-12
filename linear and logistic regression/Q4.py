import numpy as np
import pandas as pd
np.seterr(all='ignore')

class MyLogisticRegression():

  def __init__(self, alpha=0.1, n_iterations=100):
    # constructor
    self.theta = None
    self.alpha = alpha    # alpha is the learning rate, this hyperparameter is set by the programmer
    self.n_iterations = n_iterations  # number of iterations for which gradient descent is run
    np.seterr(all='ignore')

  def sigmoid(self, z):
    # sigmoid function
    return 1/(1 + np.exp(-z))   # sigmoid fn

  def log_loss(self, n_samples, h_theta_of_x, y):
    # cross entropy loss
    # returns the cost
    first_term = y*np.nan_to_num(np.log(h_theta_of_x))   # log loss
    second_term = (1-y)*np.nan_to_num(np.log(1-h_theta_of_x))
    cost = -(1/n_samples) * np.sum(first_term + second_term)
    return cost

  def batch_gradient_descent(self, X_train, y_train, n_features, n_samples):
    # implementing batch gradient descent
    for epoch in range(self.n_iterations):
      h_theta_of_x = self.sigmoid(np.dot(X_train, self.theta))  # predicted value

      d_theta = 1/(n_samples) * np.dot(X_train.T, (h_theta_of_x - y_train))
      d_theta = d_theta.reshape((n_features+1,1))

      self.theta -= self.alpha * d_theta

  def fit(self, X, y):
    #reshape y and adding 1s
    n_samples = X.shape[0]
    n_features = X.shape[1]
    temp = np.empty((n_samples, n_features+1), dtype=float)
    for i in range(n_samples):
      temp[i] = np.append(X[i], 1)
    X = temp

    self.theta = np.zeros((n_features+1,1))    # these are the model parameters
    y = y.reshape((-1,1))

    self.batch_gradient_descent(X, y, n_features, n_samples)   # BGD

    h_theta_of_x = self.sigmoid(np.dot(X, self.theta))  # predicted value
    predicted = np.around(h_theta_of_x)
    accuracy = (predicted == y).all(axis=1).mean()   # rounding off according to 0.5

    return self.theta


  def predict(self, X):
    # prediction using model
    # returns predicted value
    n_samples = X.shape[0]
    n_features = X.shape[1]
    temp = np.empty((n_samples, n_features+1), dtype=float)
    for i in range(n_samples):
      temp[i] = np.append(X[i], 1)
    X = temp
    y = self.sigmoid(np.dot(X, self.theta))
    # return the numpy array y which contains the predicted values
    # y_final = np.around(y)
    return y

df = pd.DataFrame(columns=['label', 'disease_spread', 'age'])

count = 0
with open('Q4_Dataset.txt') as file:    # file to dataframe process
  data = file.read()
data = data.split('\n')
data = data[:-1]
final = []
for elt in data:
  final.append(elt.strip())  # remove additional spaces
for row in final:
  row = row.split(' ')   # split on space
  x = []
  for elt in row:
    if elt != '':
      x.append(elt)
  df.loc[count] =  [float(elt) for elt in x]   # converting to float
  count += 1

y = df['label'].values
y = y.astype(int)

X = df.drop(['label'], axis=1)
X = X.values

logistic = MyLogisticRegression(alpha=0.001, n_iterations=100000)
theta = logistic.fit(X, y)

# Beta calculate
print('B0:', theta[2])
print('B1:', theta[0])
print('B2:', theta[1])
print('exp(B1):', np.exp(theta[0]))
print('exp(B2):', np.exp(theta[1]))

#  for last part
disease_spread = 0.75
age = 2
X_test = [[disease_spread, age]]
X_test = np.array(X_test)
y_predicted = logistic.predict(X_test)
print('predicted', y_predicted)
