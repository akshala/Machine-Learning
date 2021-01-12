import numpy as np 
import h5py
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class MyNaiveBayes():
	def __init__(self):
		self.classwise_mean_var = {}    # to store classwise mean and variance

	def separate_classes(self, X, y):   # separate data on the basis of class
		classes = {}                    # dictionary to store data corresponding to each class
		n = X.shape[0]
		for i in range(n):
			if y[i] not in classes:
				classes[y[i]] = []          # if class not in dictionary add the class
				classes[y[i]].append(X[i])  # append data
			else:
				classes[y[i]].append(X[i])  # append data
		return classes

	def mean(self, X):
		return np.sum(X)/len(X) if len(X)>0 else 0    # get mean of numpy list X

	def variance(self, X):   # to calculate variance
		mean = self.mean(X)  # get mean of X
		n = len(X) 
		sumX = 0
		for elt in X:
			sumX += (elt - mean)**2 
		var = (sumX/(n-1))
		if var == 0:          # if variance is 0 make it 1e-10
			var = 1e-10
		return var

	def get_mean_var(self, X):    # getting columnwise mean, variance and count
		df = pd.DataFrame(X)
		columnwise_mean_var = []
		for column in df:
			mean = self.mean(np.array(df[column]))     # mean of column
			var = self.variance(np.array(df[column]))  # variance of column
			count = len(df[column])                    # number of occurences of that class = length of column
			columnwise_mean_var.append((mean, var, count))
		return columnwise_mean_var

	def fit(self, X, y):
		classes = self.separate_classes(X, y)       # separate data on basis of classes
		for label, data in classes.items():
			data = np.array(data)
			columnwise_mean_var = self.get_mean_var(data)   # columnwise mean, variance and count for each class' data
			self.classwise_mean_var[label] = columnwise_mean_var

	def gaussian_probability(self, v, mean, var):   # calculating gaussian probability
		power = ((v - mean)**2)/(2 * var)
		div = np.sqrt(2 * 3.14 * var)
		result = (1/div)*np.exp(-power)
		return result

	def class_probability(self, test_row):    # getting probability for each class
		prob = {}
		total = 0
		for label in self.classwise_mean_var:               # total count of all the classes
			total += self.classwise_mean_var[label][0][2]    
		for label, stats in self.classwise_mean_var.items():
			n = len(stats)
			prob[label] = np.log(stats[0][2]/total)          # count of that particular class
			for i in range(n):
				v = test_row[i]
				gp = self.gaussian_probability(v, stats[i][0], stats[i][1])   # gaussian probability for each column
				if gp <= 0:
					continue
				prob[label] += np.log(gp)   # add probability for each class
		return prob

	def rowise_prediction(self, row):
		prob = self.class_probability(row)   # getting probability for each class
		max_prob = -9999999999
		max_prob_label = None
		for label, p in prob.items():        # get max probability and class of that max probability
			if p > max_prob:
				max_prob = p
				max_prob_label = label
		return max_prob_label

	def predict(self, X):
		pred = []
		for elt in X:
			pred.append(self.rowise_prediction(elt))   # rowise prediction for each test point
		return pred

	def accuracy(self, pred, actual):   # accuracy of the model
		n = len(pred)
		corr = 0
		for i in range(n):
			if pred[i] == actual[i]:
				corr += 1
		return corr/n

# Dataset A

hf = h5py.File('part_A_train.h5', 'r')     # read .h5 file using h5py
X = np.array(hf.get('X'))      # get X
y = np.array(hf.get('Y'))      # get y
y = np.where(y==1)[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)   # train test split of 80:20 with stratify

NB = MyNaiveBayes()             # my Gaussian Naive Bayes
NB.fit(X_train, y_train)
pred = NB.predict(X_test)
print('Dataset A')
print('Accuracy from MyNaiveBayes', NB.accuracy(pred, y_test))

clf = GaussianNB()              # sklearn Gaussian Naive Bayes
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Accuracy from sklearn', accuracy_score(y_test, pred))

# Dataset B

hf = h5py.File('part_B_train.h5', 'r')     # read .h5 file using h5py
X = np.array(hf.get('X'))      # get X
y = np.array(hf.get('Y'))      # get y
y = np.where(y==1)[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

NB = MyNaiveBayes()            # my Gaussian Naive Bayes
NB.fit(X_train, y_train)
pred = NB.predict(X_test)
print('Dataset B')
print('Accuracy from MyNaiveBayes', NB.accuracy(pred, y_test))

clf = GaussianNB()            # sklearn Gaussian Naive Bayes
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Accuracy from sklearn', accuracy_score(y_test, pred))