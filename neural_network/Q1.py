import numpy as np 
from copy import deepcopy
import matplotlib.pyplot as plt
np.random.seed(42)
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')

        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.W = []
        self.b = []

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.maximum(0, X)

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.where(X<=0, 0, 1)

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1 + np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        s = self.sigmoid(X)
        return s * (1-s)

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        t = self.tanh(X)
        return 1 - t**2

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        numerator = np.exp(X - np.max(X, axis=0))
        denominator = np.sum(numerator, axis=0)
        return numerator/denominator

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        s = self.softmax(X)
        s = s.reshape((-1, 1))
        return np.diagflat(s) - np.dot(s, s.T)

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return 0.01 * np.random.rand(shape)

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return 0.01 * np.random.normal(size=shape)

    def param_init(self):    
    # initialising parameters for each layer
        for i in range(1, self.n_layers):
            if self.weight_init == 'zero':                     # zero initialisation
                self.W.append(self.zero_init((self.layer_sizes[i], self.layer_sizes[i-1])))
                self.b.append(self.zero_init((self.layer_sizes[i], 1)))
            elif self.weight_init == 'random':                 # random initialisation
                self.W.append(self.random_init((self.layer_sizes[i], self.layer_sizes[i-1])))
                self.b.append(self.random_init((self.layer_sizes[i], 1)))
            elif self.weight_init == 'normal':                 # normal initialisation
                self.W.append(self.normal_init((self.layer_sizes[i], self.layer_sizes[i-1])))
                self.b.append(self.normal_init((self.layer_sizes[i], 1)))

    def forward_propogation(self, X):
        """
        forward propogation in the model

        Parameters
        ----------
        X : 2-dimensional numpy array which contains the input data

        Returns
        -------
        y_pred : 1-dimensional numpy array which contains the predicted probabilities for each class
        hidden_layer_vals : tuple containing z, A and W weight
        """
        A = X.T
        hidden_layer_vals = [[X.T]]
        for i in range(self.n_layers-2):
            A_prev = A
            # print('A', A.shape, 'W', self.W[i].shape, 'b', self.b[i].shape)
            z = np.dot(self.W[i], A_prev) + self.b[i]
            if self.activation == 'relu':
                A = self.relu(z)
            elif self.activation == 'sigmoid':
                A = self.sigmoid(z)
            elif self.activation == 'linear':
                A = self.linear(z)
            elif self.activation == 'tanh':
                A = self.tanh(z)
            elif self.activation == 'softmax':
                A = self.softmax(z)
            to_save_val = (z, A, self.W[i])
            hidden_layer_vals.append(to_save_val)

        z = np.dot(self.W[-1], A) + self.b[-1]
        y_pred = self.softmax(z)
        to_save_val = (z, y_pred, self.W[-1])
        hidden_layer_vals.append(to_save_val)

        return y_pred, hidden_layer_vals

    def gradient_norm(self,X):
        """
        gradient normalisation

        Parameters
        ----------
        X : 2-dimensional numpy array which contains the gradient

        Returns
        -------
        normalised gradient
        """
        norm = np.maximum(1, np.linalg.norm(X, axis=0, keepdims=True))
        return X/norm

    def backward_propogation(self, hidden_layer_vals, y):
        """
        forward propogation in the model

        Parameters
        ----------
        hidden_layer_vals : tuple which is the output of forward pass
        y : actual classes

        Returns
        -------
        weight updates
        """
        dJ_dW = []
        dJ_db = []
        all_delta = {}
        n_samples = y.shape[0]

        delta = hidden_layer_vals[-1][0] - y.T
        delta = self.gradient_norm(delta)
        all_delta[self.n_layers-1] = delta

        dJ_dW.append((1/n_samples) * np.dot(delta, hidden_layer_vals[-2][0].T))
        dJ_db.append((1/n_samples) * np.sum(delta, axis=1, keepdims=True))

        for i in range(self.n_layers-2, 0, -1):
            if self.activation == 'relu':
                delta_act = self.relu_grad(hidden_layer_vals[i][1])
            elif self.activation == 'sigmoid':
                delta_act = self.sigmoid_grad(hidden_layer_vals[i][1])
            elif self.activation == 'linear':
                delta_act = self.linear_grad(hidden_layer_vals[i][1])
            elif self.activation == 'tanh':
                delta_act = self.tanh_grad(hidden_layer_vals[i][1])
            elif self.activation == 'softmax':
                delta_act = self.softmax_grad(hidden_layer_vals[i][1])

            delta = np.dot(hidden_layer_vals[i+1][2].T, all_delta[i+1])
            delta = np.multiply(delta, delta_act)
            delta = self.gradient_norm(delta)
            all_delta[i] = delta

            dJ_dW.append((1/n_samples) * np.dot(delta, hidden_layer_vals[i-1][0].T))
            dJ_db.append((1/n_samples) * np.sum(delta, axis=1, keepdims=True))

        return dJ_dW[::-1], dJ_db[::-1]

    def update_param(self, dJ_dW, dJ_db):
        """
        forward propogation in the model

        Parameters
        ----------
        dJ_dW : updates for W
        dJ_db : updates for b
        """
        for i in range(self.n_layers-1):
            self.W[i] -= self.learning_rate * dJ_dW[i]
            self.b[i] -= self.learning_rate * dJ_db[i]

    def one_hot_encoding(self, y):
        """
        forward propogation in the model

        Parameters
        ----------
        y : 1D numpy array to be encoded
        Returns
        -------
        y_encoded : one hot encoded y
        """
        y_encoded = np.zeros((y.size, y.max()+1))
        y_encoded[np.arange(y.size), y] = 1
        return y_encoded

    def fit(self, X, y, X_test, y_test):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        # fit function has to return an instance of itself or else it won't work with test.py

        y_encoded = self.one_hot_encoding(y)
        self.param_init(X.shape[1], y_encoded.shape[1])
        n_samples = X.shape[0]
        num_batches = n_samples//self.batch_size
        train_loss = []
        test_loss = []
        epochs = []

        for epoch in range(self.num_epochs):
            loss = 0
            for i in range(num_batches):

                X_batch = X[i*self.batch_size: (i+1)*self.batch_size, :]
                y_batch = y_encoded[i*self.batch_size: (i+1)*self.batch_size, :]

                y_pred, hidden_layer_vals = self.forward_propogation(X_batch)

                dJ_dW, dJ_db = self.backward_propogation(hidden_layer_vals, y_batch)
                self.update_param(dJ_dW, dJ_db)

            y_pred_train, _ = self.forward_propogation(X)
            y_pred_test, hidden_layer_vals = self.forward_propogation(X_test)

            epochs.append(epoch)
            train_loss.append(self.cost(y_pred_train, y))
            test_loss.append(self.cost(y_pred_test, y_test))

        y_pred_test, hidden_layer_vals = self.forward_propogation(X_test)
        self.plot_tsne(hidden_layer_vals, y_test)

        plt.plot(epochs, train_loss, label='train') 
        plt.plot(epochs, test_loss, label='test') 
        plt.xlabel('Number of epochs') 
        plt.ylabel('Error') 
        plt.title('error vs epochs') 
        plt.legend()
        plt.show() 

        return self

    def cost(self, predicted, y):
        """
        Cross entropy loss for multiclass classification

        Parameters
        ----------
        predicted : 2-dimensional numpy array of shape (n_classes, n_samples) which contains probabilities for each class

        y : 1-dimensional numpy array of shape (n_samples,) which contains actual classes
        
        Returns
        -------
        loss
        """
        loss = 0
        predicted = predicted.T
        n = predicted.shape[1]
        for i in range(n):
            loss -= np.log(predicted[i][y[i]])
        return loss/n

    def plot_tsne(self, hidden_layer_vals, y_test):
        """
        Plotting TSNE graph

        Parameters
        ----------
        hidden_layer_vals : 2D numpy array containing output of hidden layers

        y_test : actual classes
        """
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        train_data = hidden_layer_vals[-2][1].T
        tsne_results = tsne.fit_transform(train_data)         # fit t-SNE on train data

        features = ['num' + str(i) for i in range(train_data.shape[1])]
        df = pd.DataFrame(train_data, columns=features)          # new dataframe with X and y values
        df['y'] = y_test

        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]
        plt.figure(figsize=(8,8))
        sns.scatterplot(                        # plotting t-SNE graph
            x='tsne-2d-one', y='tsne-2d-two',
            hue='y',
            palette=sns.color_palette('hls', 10),
            data=df,
            legend='full',
            alpha=0.3
        )
        plt.show()

        
    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

        y, hidden_layer_vals = self.forward_propogation(X)

        # return the numpy array y which contains the predicted values
        return hidden_layer_vals[-1][0].T

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values
        probability = self.predict_proba(X)
        return np.argmax(probability, axis=1)

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        # return the numpy array y which contains the predicted values
        y_pred = self.predict(X)
        # print(y, y_pred)
        actual = np.count_nonzero(y == y_pred)
        # print(actual, len(y))
        return actual / len(y)

