from Q1 import MyNeuralNetwork
from keras.datasets import mnist
import numpy as np
from numpy import save
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

mean = np.mean(X_train, axis=0)
variance = np.std(X_train, axis=0)
X_train = (X_train - mean)/(variance + 1e-8)
X_test = (X_test - mean)/(variance + 1e-8)

nn = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'relu', 0.1, 'normal', 200, 100)
nn.fit(X_train, y_train, X_test, y_test)
save('Q1_relu_W.npy', nn.W)
save('Q1_relu_b.npy', nn.b)
print('Train accuracy:', nn.score(X_train, y_train))
print('Test accuracy:', nn.score(X_test, y_test))

nn = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'sigmoid', 0.1, 'normal', 100, 100)
nn.fit(X_train, y_train, X_test, y_test)
save('Q1_sigmoid_W.npy', nn.W)
save('Q1_sigmoid_b.npy', nn.b)
print('Train accuracy:', nn.score(X_train, y_train))
print('Test accuracy:', nn.score(X_test, y_test))

nn = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'linear', 0.1, 'normal', 200, 100)
nn.fit(X_train, y_train, X_test, y_test)
save('Q1_linear_W.npy', nn.W)
save('Q1_linear_b.npy', nn.b)
print('Train accuracy:', nn.score(X_train, y_train))
print('Test accuracy:', nn.score(X_test, y_test))

nn = MyNeuralNetwork(5, [784, 256, 128, 64, 10], 'tanh', 0.1, 'normal', 200, 100)
nn.fit(X_train, y_train, X_test, y_test)
save('Q1_tanh_W.npy', nn.W)
save('Q1_tanh_b.npy', nn.b)
print('Train accuracy:', nn.score(X_train, y_train))
print('Test accuracy:', nn.score(X_test, y_test))

model = MLPClassifier(activation='relu', learning_rate_init=0.1, max_iter=100, hidden_layer_sizes=(256, 128, 64), random_state=42, solver='sgd', alpha=0, batch_size=200, tol = 0, shuffle = False, learning_rate = "constant", momentum = 0)
model.fit(X_train, y_train)
print('Train accuracy: ', model.score(X_train, y_train))
print('Test accuracy: ', model.score(X_test, y_test))

model = MLPClassifier(activation='logistic', learning_rate_init=0.1, max_iter=100, hidden_layer_sizes=(256, 128, 64), random_state=42, solver='sgd', alpha=0, batch_size=100, tol = 0, shuffle = False, learning_rate = "constant", momentum = 0)
model.fit(X_train, y_train)
print('Train accuracy: ', model.score(X_train, y_train))
print('Test accuracy: ', model.score(X_test, y_test))

model = MLPClassifier(activation='identity', learning_rate_init=0.1, max_iter=100, hidden_layer_sizes=(256, 128, 64), random_state=42, solver='sgd', alpha=0, batch_size=200, tol = 0, shuffle = False, learning_rate = "constant", momentum = 0)
model.fit(X_train, y_train)
print('Train accuracy: ', model.score(X_train, y_train))
print('Test accuracy: ', model.score(X_test, y_test))

model = MLPClassifier(activation='tanh', learning_rate_init=0.1, max_iter=100, hidden_layer_sizes=(256, 128, 64), random_state=42, solver='sgd', alpha=0, batch_size=200, tol = 0, shuffle = False, learning_rate = "constant", momentum = 0)
model.fit(X_train, y_train)
print('Train accuracy: ', model.score(X_train, y_train))
print('Test accuracy: ', model.score(X_test, y_test))
