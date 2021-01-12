import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset
            df = pd.DataFrame(columns=['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])
            count = 0

            with open('Dataset.data') as file:  # reading data from file
                data = file.read()

            data = data.split('\n')   # split data into different rows
            data = data[:-1]   # last one is empty
            for row in data:
                row = row.split()
                df.loc[count] = row   # add in dataframe
                count += 1

            df['M'] = np.where(df.sex=='M', 1,0)   # genders are turned to a one hot encoding
            df['F'] = np.where(df.sex=='F', 1,0)
            df['I'] = np.where(df.sex=='I', 1,0)
            df = df.drop(['sex'], axis=1)
            df = df.dropna()

            df = df.sample(frac=1).reset_index(drop=True)    # shuffle dataframe

            X = df.drop(['rings'], axis=1)
            X = X.values
            X = X.astype(float)
            y = df['rings'].values
            y = y.astype(float)

        elif dataset == 1:
            # Implement for the video game dataset
            df = pd.read_csv('VideoGameDataset - Video_Games_Sales_as_at_22_Dec_2016.csv')  # read csv directly into a dataframe
            df1 = df[['Critic_Score', 'User_Score', 'Global_Sales']]
            df1 = df1.dropna()
            df1 = df1[df1.User_Score != 'tbd']

            df1 = df1.sample(frac=1).reset_index(drop=True)   # shuffle rows

            X = df1.drop(['Global_Sales'], axis=1)
            X = X.values
            X = X.astype(float)
            y = df1['Global_Sales'].values
            y = y.astype(float)

        elif dataset == 2:
            # Implement for the banknote authentication dataset
            df = pd.DataFrame(columns=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
            count = 0

            with open('data_banknote_authentication.txt') as file:   # reading file 
                data = file.read()
            data = data.split('\n')
            data = data[:-1]
            for row in data:
                row = row.split(',')
                df.loc[count] = [float(elt) for elt in row[:-1]] + [int(row[-1])]   # last column has class so it is int rest are float
                count += 1

            df = df.sample(frac=1).reset_index(drop=True)   # shuffle dataset

            X = df.drop(['class'], axis=1)
            X = X.values
            y = df['class'].values
            y = y.astype(int)

        return X, y

class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """

    def __init__(self, alpha=0.01, n_iterations=1000, cost_fn='MAE'):
        # constructor
        self.theta = None    # model parameter
        self.alpha = alpha    # alpha is the learning rate, this hyperparameter is set by the programmer
        self.n_iterations = n_iterations  # number of iterations for which gradient descent is run
        self.cost_fn = cost_fn 
        self.train_history = None  # to store values of theta in all
        self.val_history = None 

    def MAE(self, n_samples, h_theta_of_x, y):
        # implemanting MAE loss
        # returns the loss
        cost = (1/n_samples) * (np.sum(np.absolute(h_theta_of_x - y)))    # MAE loss calculated
        return cost

    def RMSE(self, n_samples, h_theta_of_x, y):
        # implementing RMSE loss
        # return the loss
        cost = (1/n_samples**0.5) * (np.sum((h_theta_of_x - y)**2)) ** 0.5   # RMSE loss calculated
        return cost

    def gradient_descent_mae(self, X_train, y_train, X_val, y_val):
        # implementing gradient descent for MAE
        n_samples = X_train.shape[0]

        for epoch in range(self.n_iterations):
            h_theta_of_x = np.dot(X_train, self.theta)   # predicted value of y
            train_cost = self.MAE(n_samples, h_theta_of_x, y_train)   # MAE loss
            self.train_history[epoch] = train_cost   # storing loss in array

            try: # if validation set is present
                if X_val.all() != None and y_val.all() != None:
                    val_n_samples = X_val.shape[0]
                    val_h_theta_of_x = np.dot(X_val, self.theta)
                    val_cost = self.MAE(val_n_samples, val_h_theta_of_x, y_val)   # MAE loss
                    self.val_history[epoch] = val_cost
            except AttributeError:
                pass

            signed_matrix = np.dot(X_train.T, (np.sign(h_theta_of_x - y_train)))   # derivative would be sign matrix for predicted-actual multiplied by X.T
            d_theta = (1/n_samples) * signed_matrix
            self.theta -= self.alpha * d_theta   # updating theta

    def gradient_descent_rmse(self, X_train, y_train, X_val, y_val):
        # implementing gradient descent for RMSE
        n_samples = X_train.shape[0]

        for epoch in range(self.n_iterations):
            h_theta_of_x = np.dot(X_train, self.theta) # predicted value of y
            train_cost = self.RMSE(n_samples, h_theta_of_x, y_train)   # MAE loss
            self.train_history[epoch] = train_cost   # storing loss in array

            try: # if validation set is present
                if X_val.all() != None and y_val.all() != None:
                    val_n_samples = X_val.shape[0]
                    val_h_theta_of_x = np.dot(X_val, self.theta)
                    val_cost = self.RMSE(val_n_samples, val_h_theta_of_x, y_val)   # MAE loss
                    self.val_history[epoch] = val_cost
            except AttributeError:
                pass

            first = ((1/np.sum(np.square(h_theta_of_x - y_train)))**0.5)  # derivative for RMSE loss
            second = np.dot(X_train.T, (h_theta_of_x - y_train))

            d_theta = (1/(n_samples**0.5)) * first * second
            self.theta = self.theta - d_theta * self.alpha   # updating theta

    def fit(self, X, y, X_val=None, y_val=None):
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
        #Adding an extra column of 1s for constant term
        n_samples = X.shape[0]
        n_features = X.shape[1]
        temp = np.empty((n_samples, n_features+1), dtype=float)   # adding extra 1s to X
        for i in range(n_samples):
          temp[i] = np.append(X[i], 1)
        X = temp
        y = y.reshape((-1,1))   # reshaping y 

        try:    # if validation set is present
            if X_val.all() != None and y_val.all() != None:
            # adding extra 1s and reshaping y
                val_n_samples = X_val.shape[0]
                val_n_features = X_val.shape[1]
                temp = np.empty((val_n_samples, val_n_features+1), dtype=float)
                for i in range(val_n_samples):
                    temp[i] = np.append(X_val[i], 1)
                X_val = temp
                y_val = y_val.reshape((-1,1))
        except AttributeError:
            pass

        self.theta = np.zeros((n_features+1,1))    # these are the model parameters
        self.train_history = np.zeros(self.n_iterations)   # initialising array store train loss history
        self.val_history = np.zeros(self.n_iterations)  # initialising array store val loss history
        
        if self.cost_fn == 'RMSE':
          self.gradient_descent_rmse(X, y, X_val, y_val)   # RMSE gradient call
          h_theta_of_x = np.dot(X, self.theta)
          cost = self.RMSE(n_samples, h_theta_of_x, y)  # final RMSE loss
          # print('training_loss', cost)

        elif self.cost_fn == 'MAE':
          self.gradient_descent_mae(X, y, X_val, y_val)  # MAE gradient call
          h_theta_of_x = np.dot(X, self.theta)
          cost = self.MAE(n_samples, h_theta_of_x, y)  # final MSE loss
          # print('training_loss', cost)


        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def kFold(self, k, X, y):
        # K fold CV
        n_samples = X.shape[0]
        n_features = X.shape[1]
        data_folds = []
        step_size = n_samples//k

        # dividing into k parts
        for start in range(k):
          tup = (X[start*step_size:(start*step_size) + step_size], y[start*step_size:(start*step_size) + step_size])
          data_folds.append(tup)

        costs = []
        train_total_cost = np.zeros(self.n_iterations)
        val_total_cost = np.zeros(self.n_iterations)
        
        for curr_fold in range(k):
            # assigning test fold
            test_X = data_folds[curr_fold][0]
            test_y = data_folds[curr_fold][1]

            if curr_fold == 0:
                # if test fold is 0 then train has to start from 1
                train_X = data_folds[1][0]
                train_y = data_folds[1][1] 
            else: 
                train_X = data_folds[0][0]
                train_y = data_folds[0][1]

            for i in range(1, k):
                # rest of train fold
                if i != curr_fold:
                  train_X = np.concatenate((train_X, data_folds[i][0]))
                  train_y = np.concatenate((train_y, data_folds[i][1]))

            linear = self.fit(train_X, train_y, test_X, test_y)

            n_samples_test = test_X.shape[0]
            n_features_test = test_X.shape[1]

            # adding extra 1s to X and reshaping y
            temp = np.empty((n_samples_test, n_features_test+1), dtype=float)
            for i in range(n_samples_test):
                temp[i] = np.append(test_X[i], 1)
            test_X = temp
            test_y = test_y.reshape((-1,1))

            # adding extra 1s to X and reshaping y
            n_samples_train = train_X.shape[0]
            n_features_train = train_X.shape[1]

            temp = np.empty((n_samples_train, n_features_test+1), dtype=float)
            for i in range(n_samples_train):
                temp[i] = np.append(train_X[i], 1)
            train_X = temp
            train_y = train_y.reshape((-1,1))

            if self.cost_fn == "MAE":

                h_theta_of_x = np.dot(train_X, linear.theta)
                train_error = self.MAE(n_samples_train, h_theta_of_x, train_y)  # train error

                h_theta_of_x = np.dot(test_X, linear.theta)
                test_error = self.MAE(n_samples_test, h_theta_of_x, test_y)  # val error

                print('fold', curr_fold, 'train', train_error, 'test', test_error)  # for best

                train_total_cost = np.add(train_total_cost, linear.train_history)  # adding up errors of each folds
                val_total_cost = np.add(val_total_cost, linear.val_history)
            

            elif self.cost_fn == "RMSE":

                h_theta_of_x = np.dot(train_X, linear.theta)
                train_error = self.MAE(n_samples_train, h_theta_of_x, train_y)   # train error

                h_theta_of_x = np.dot(test_X, linear.theta)  
                test_error = self.MAE(n_samples_test, h_theta_of_x, test_y)    # val error

                print('fold', curr_fold, 'train', train_error, 'test', test_error)   # for best

                train_total_cost = np.add(train_total_cost, linear.train_history)   # adding up errors of each folds
                val_total_cost = np.add(val_total_cost, linear.val_history)

        # taking average error
        train_total_cost /= k
        val_total_cost /= k
        
        plt.plot(train_total_cost, label='train')   # graphs
        plt.plot(val_total_cost, label='val')
        plt.legend()
        plt.show()


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
        n_samples = X.shape[0]
        n_features = X.shape[1]
        temp = np.empty((n_samples, n_features+1), dtype=float)
        for i in range(n_samples):
          temp[i] = np.append(X[i], 1)
        X = temp
        y = np.dot(X, self.theta)
        # return the numpy array y which contains the predicted values
        return y

    def normal_equation(self, X, y):
        # adding extra 1s to X reshaping y
        # returns the theta values obtained using normal equation
        n_samples = X.shape[0]
        n_features = X.shape[1]
        temp = np.empty((n_samples, n_features+1), dtype=float)
        for i in range(n_samples):
          temp[i] = np.append(X[i], 1)
        X = temp
        y = y.reshape((-1,1))
        first = np.dot(X.T, X)   # calculating using normal eq
        second = np.linalg.pinv(first)
        third = np.dot(X.T, y)
        theta = np.dot(second, third)
        print(theta)
        return theta

    def loss_using_norm(self, X, y, k):
        # K fold CV
        n_samples = X.shape[0]
        n_features = X.shape[1]
        data_folds = []
        step_size = n_samples//k

        # dividing into k parts
        for start in range(k):
          tup = (X[start*step_size:(start*step_size) + step_size], y[start*step_size:(start*step_size) + step_size])
          data_folds.append(tup)

        # according to best fold
        test_X = data_folds[2][0]
        test_y = data_folds[2][1]
          # according to best fold
        train_X = data_folds[0][0]
        train_y = data_folds[0][1] 
          # according to best fold
        train_X = np.concatenate((train_X, data_folds[1][0]))
        train_y = np.concatenate((train_y, data_folds[1][1]))

        n_samples_test = test_X.shape[0]
        n_features_test = test_X.shape[1]

        # adding extra 1s to X and reshaping y
        temp = np.empty((n_samples_test, n_features_test+1), dtype=float)
        for i in range(n_samples_test):
            temp[i] = np.append(test_X[i], 1)
        test_X = temp
        test_y = test_y.reshape((-1,1))

        # adding extra 1s to X and reshaping y
        n_samples_train = train_X.shape[0]
        n_features_train = train_X.shape[1]

        temp = np.empty((n_samples_train, n_features_test+1), dtype=float)
        for i in range(n_samples_train):
            temp[i] = np.append(train_X[i], 1)
        train_X = temp
        train_y = train_y.reshape((-1,1))

        theta = self.normal_equation(X, y)   # theta using normal eq

        h_theta_of_x = np.dot(train_X, theta)
        train_error = self.MAE(n_samples_train, h_theta_of_x, train_y)   # MAE error
        print('train', train_error)

        h_theta_of_x = np.dot(test_X, theta)
        val_error = self.MAE(n_samples_test, h_theta_of_x, test_y)    # RMSE error
        print('val', val_error)



class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self, alpha=0.1, n_iterations=100, cost_fn='Batch'):
        self.theta = None     # model parameters
        self.alpha = alpha    # alpha is the learning rate, this hyperparameter is set by the programmer
        self.n_iterations = n_iterations  # number of iterations for which gradient descent is run
        self.cost_fn = cost_fn
        np.seterr(all='ignore')
        self.train_history = None  # to store values of theta in all
        self.val_history = None 

    def sigmoid(self, z):  
     # sigmoid fn
        return 1/(1 + np.exp(-z))

    def log_loss(self, n_samples, h_theta_of_x, y):     
    # log loss funtion
        minimum = np.full((n_samples, 1), 0.0000001)
        first_term = y*np.nan_to_num(np.log(np.maximum(minimum, h_theta_of_x)))
        second_term = (1-y)*np.nan_to_num(np.log(np.maximum(minimum, 1-h_theta_of_x)))
        cost = -(1/n_samples) * np.sum(first_term + second_term)
        return cost

    def batch_gradient_descent(self, X_train, y_train, X_val, y_val):
        # implementing batch gradient descent
        n_samples = X_train.shape[0]
        for epoch in range(self.n_iterations):
            h_theta_of_x = self.sigmoid(np.dot(X_train, self.theta))   # predicted value of y
            train_cost = self.log_loss(n_samples, h_theta_of_x, y_train)   # train loss
            self.train_history[epoch] = train_cost    # storing train loss

            try:  # if validation set is present
                if X_val.all() != None and y_val.all() != None:
                    val_n_samples = X_val.shape[0]
                    val_h_theta_of_x = np.dot(X_val, self.theta)
                    val_cost = self.log_loss(val_n_samples, val_h_theta_of_x, y_val)   # val loss
                    self.val_history[epoch] = val_cost
            except AttributeError:
                pass

            d_theta = 1/(n_samples) * np.dot(X_train.T, (h_theta_of_x - y_train))  # derivative
            self.theta -= self.alpha * d_theta

        plt.plot(self.train_history, label='Train loss')

        try: # if validation set is present
          if X_val.all() != None and y_val.all() != None:
            plt.plot(self.val_history, label='Val loss')   # val loss
        except AttributeError:
          pass
        plt.legend()
        plt.show()

    def stochastic_gradient_descent(self, X_train, y_train, X_val, y_val):
        # Implementing stochastic gradient descent
        n_samples = X_train.shape[0]
        for epoch in range(self.n_iterations):

            for i in range(n_samples):
                # updating theta on each train point
                curr_X = X_train[i]
                curr_y = y_train[i]
                curr_X = curr_X.reshape((curr_X.shape[0], 1))
                curr_y = curr_y.reshape((curr_y.shape[0], 1))
                curr_h_theta_of_x = self.sigmoid(np.dot(curr_X.T, self.theta))  # predicted y
                d_theta = np.dot(curr_X, (curr_h_theta_of_x - curr_y))   # derivative
                self.theta -= self.alpha * d_theta     # update

            h_theta_of_x = self.sigmoid(np.dot(X_train, self.theta))

            train_cost = self.log_loss(n_samples, h_theta_of_x, curr_y)   # training loss
            self.train_history[epoch] = train_cost

            try: # if validation set is present
                if X_val.all() != None and y_val.all() != None:
                    val_n_samples = X_val.shape[0]
                    val_h_theta_of_x = np.dot(X_val, self.theta)
                    val_cost = self.log_loss(val_n_samples, val_h_theta_of_x, y_val)
                    self.val_history[epoch] = val_cost
            except AttributeError:
                pass


        plt.plot(self.train_history, label='Train loss')
        try: # if validation set is present
            if X_val.all() != None and y_val.all() != None:
                plt.plot(self.val_history, label='Val loss')
        except AttributeError:
            pass
        plt.legend()
        plt.show()

    def train_val_test_split(self, X, y):
        # splitting into training validation and test
        # 7:1:2 spit
        n_samples = X.shape[0]
        n_features = X.shape[1]
        total_divisions = 10
        step_size = n_samples//10
        start = 0

        # train set 7/10
        X_train = X[start:start + 7*step_size]
        y_train = y[start:start + 7*step_size]
        y_train = y_train.reshape((-1,1))
        start += 7*step_size

        # val set 1/10
        X_val = X[start:start + 1*step_size]
        y_val = y[start:start + 1*step_size]
        y_val = y_val.reshape((-1,1))
        start += 1*step_size

        # test set 2/10
        X_test = X[start:start + 2*step_size]
        y_test = y[start:start + 2*step_size]
        y_test = y_test.reshape((-1,1))

        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        # fit function has to return an instance of itself or else it won't work with test.py
        # adding extra 1s and reshaping y
        n_samples = X.shape[0]
        n_features = X.shape[1]
        temp = np.empty((n_samples, n_features+1), dtype=float)
        for i in range(n_samples):
          temp[i] = np.append(X[i], 1)
        X = temp

        try:    # if validation set is present
          if X_val.all() != None and y_val.all() != None:
            # adding extra 1s and reshaping y
            val_n_samples = X_val.shape[0]
            val_n_features = X_val.shape[1]
            temp = np.empty((val_n_samples, val_n_features+1), dtype=float)
            for i in range(val_n_samples):
              temp[i] = np.append(X_val[i], 1)
            X_val = temp
        except AttributeError:
          pass

        self.theta = np.zeros((n_features+1,1))    # these are the model parameters
        self.train_history = np.zeros(self.n_iterations)
        self.val_history = np.zeros(self.n_iterations)
        y = y.reshape((-1,1))

        if self.cost_fn == 'Batch':
            self.batch_gradient_descent(X, y, X_val, y_val)  # Batch
        elif self.cost_fn == 'Stochastic':
            self.stochastic_gradient_descent(X, y, X_val, y_val)   # Stochastic

        h_theta_of_x = self.sigmoid(np.dot(X, self.theta))   # predicted value
        predicted = np.around(h_theta_of_x)   # rounding off according to 0.5
        accuracy = (predicted == y).all(axis=1).mean()   # calculating accuracy of model
        train_cost = self.log_loss(n_samples, h_theta_of_x, y)

        print('training loss', train_cost)
        print('training accuracy:', accuracy)

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        # return the numpy array y which contains the predicted values

        # adding extra 1s and reshaping y
        n_samples = X.shape[0]
        n_features = X.shape[1]
        temp = np.empty((n_samples, n_features+1), dtype=float)
        for i in range(n_samples):
          temp[i] = np.append(X[i], 1)
        X = temp
        y = self.sigmoid(np.dot(X, self.theta))   # predicted value
        # return the numpy array y which contains the predicted values
        y_final = np.around(y)
        return y_final