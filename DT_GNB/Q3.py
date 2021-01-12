import h5py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import itertools
from statistics import harmonic_mean
import matplotlib.pyplot as plt
import pickle

def get_accuracy(pred, actual):    # getting accuracy of model using actual and predicted
	n = len(pred)
	corr = 0
	for i in range(n):
		if pred[i] == actual[i]:
			corr += 1
	return corr/n

def run_model_NGB(X, y):           # run Gaussian Naive Bayes with k fold cross validation
	clf = GaussianNB()
	folds = kFold(4, X, y)         # k fold with 4 folds
	val_accuracies = []            # validation accuracy
	train_accuracies = []          # training accuracy

	for fold in folds:
		train_X = fold[0]
		train_y = fold[1]
		test_X = fold[2]
		test_y = fold[3]
		clf.fit(train_X, train_y)  # training the model in each fold
		pred = clf.predict(test_X)
		train_accuracies.append(clf.score(train_X, train_y))
		val_accuracies.append(get_accuracy(pred, test_y))

	train_accuracy = sum(train_accuracies)/len(train_accuracies)   # train accuracy 
	val_accuracy = sum(val_accuracies)/len(val_accuracies)         #validation accuracy
	print('Train accuracy', train_accuracy)
	print('Val accuracy', val_accuracy)

def run_model_DT(param, X, y):             # run Decision Tree with k fold cross validation
	clf = DecisionTreeClassifier(**param)  # model with given parameters
	folds = kFold(4, X, y)                 # k fold cross validation with 4 folds
	val_accuracies = []
	train_accuracies = []

	for fold in folds:
		train_X = fold[0]
		train_y = fold[1]
		test_X = fold[2]
		test_y = fold[3]
		clf.fit(train_X, train_y)         # training model for each fold
		pred = clf.predict(test_X)
		train_accuracies.append(clf.score(train_X, train_y))   
		val_accuracies.append(get_accuracy(pred, test_y))      

	train_accuracy = sum(train_accuracies)/len(train_accuracies)  # training accuracy
	val_accuracy = sum(val_accuracies)/len(val_accuracies)        # validation accuracy

	return [param, val_accuracy, train_accuracy]
 
def GridSearch(parameters, X, y):      # grid search
	results = pd.DataFrame(columns=['parameters', 'val_accuracy', 'train_accuracy'])
	keys, values = zip(*parameters.items())
	index = 0
	for param_val in itertools.product(*values):   # for all possible combinations of the parameters
		curr_param = dict(zip(keys, param_val))
		param_accuracy = run_model_DT(curr_param, X, y)   # run model
		results.loc[index] = param_accuracy               # add to dataframe
		index += 1
	results.sort_values('val_accuracy', inplace = True, ascending = False)   # sort dataframe on basis of validation accuracy
	results.reset_index(inplace = True)
	return results


def kFold(k, X, y):            # K fold CV
	n_samples = X.shape[0]
	n_features = X.shape[1]
	data_folds = []
	step_size = n_samples//k

	for start in range(k):      # dividing into k parts
		tup = (X[start*step_size:(start*step_size) + step_size], y[start*step_size:(start*step_size) + step_size])
		data_folds.append(tup)

	folds = []

	for curr_fold in range(k):          # assigning test fold
		test_X = data_folds[curr_fold][0]
		test_y = data_folds[curr_fold][1]

		if curr_fold == 0:              # if test fold is 0 then train has to start from 1
			train_X = data_folds[1][0]
			train_y = data_folds[1][1] 
		else: 
			train_X = data_folds[0][0]
			train_y = data_folds[0][1]

		for i in range(1, k):           # rest of train fold
			if i != curr_fold:
				train_X = np.concatenate((train_X, data_folds[i][0]))
				train_y = np.concatenate((train_y, data_folds[i][1]))
		folds.append([train_X, train_y, test_X, test_y])
	return folds                        # return all folds

def roc(probability, y, classes):       # plotting ROC curve
	num_classes = len(classes)
	threshold = list(range(0,102))
	threshold = [t/100 for t in threshold]   # list of thresholds
	n = len(y)

	for curr_class in range(num_classes):    # ploitting ROC curve for each class
		TPR_list = []                        # True positive rates list
		FPR_list = []                        # False positive rates list
		for thresh in threshold:
			y_pred = np.zeros(y.shape[0])
			df = [[0,0], [0,0]]
			for curr_elt in range(n):        # for each test point
				if probability[curr_elt][curr_class] >= thresh:    # if probability of current class greater than threshold
					y_pred[curr_elt] = 1                           # assign that class
				else:
					y_pred[curr_elt] = 0                           # else other class
				temp = 0
				if y[curr_elt] == curr_class:
					temp = 1
				df[temp][int(y_pred[curr_elt])] += 1               # create confusion matrix

			TP = df[1][1]    # true positive
			TN = df[0][0]    # true negative
			FP = df[0][1]    # false positive
			FN = df[1][0]    # false negative
			TPR = TP/(TP + FN)     # true postive rate
			FPR = FP/ (FP + TN)    # false positive rate
			TPR_list.append(TPR)
			FPR_list.append(FPR)
		plt.plot(FPR_list, TPR_list, label='Class ' + str(curr_class))   # plot ROC curve
	plt.xlabel("False Positive Rate", fontsize=12)
	plt.ylabel("True Positive Rate", fontsize=12)
	plt.legend()
	plt.show()


def evaulation_metric(pred, actual, classes, model, X):
	accuracy = get_accuracy(pred, actual)    # accuracy of model
	print('accuracy:', accuracy)
	df = pd.DataFrame(0, columns = classes, index = classes)   # cponfusion matrix
	n = len(actual)
	for i in range(n):
		try:
			df[actual[i]][pred[i]] += 1
		except KeyError:
			pass
	print('Confusion Matrix')
	print(df)
	if len(classes) == 2:  # only 2 classes
		TP = df[1][1]      # true positive
		TN = df[0][0]      # true negative
		FP = df[0][1]      # false positive
		FN = df[1][0]      # false negative
		precision = TP/(TP + FP)                   #precision
		recall = TP/(TP + FN)                      # recall
		f1 = harmonic_mean([precision, recall])    # F1 score
		print('Precision:', precision)
		print('Recall:', recall)
		print('F1 score:', f1)
		roc(model.predict_proba(X), actual, classes)   # roc curve
	else:                         # more than 2 classes
		print('Macro average')
		precision = {}
		recall = {}
		predicted_sum = df.sum(axis=0)   # sum of predicted
		actual_sum = df.sum(axis=1)      # sum of actual
		for c in classes:
			precision[c] = [df[c][c]/predicted_sum[c]]    # classwise precision
			recall[c] = [df[c][c]/actual_sum[c]]          # classwise recall
		f1 = {}
		for c in classes: 
			f1[c] = [harmonic_mean([precision[c][0], recall[c][0]])]   # classwise f1 score

		df_precision = pd.DataFrame.from_dict(precision) 
		df_recall = pd.DataFrame.from_dict(recall) 
		df_f1 = pd.DataFrame.from_dict(f1) 
		macro_precision = df_precision.stack().mean()   # macro precisiom
		macro_recall = df_recall.stack().mean()         # macro recall
		macro_f1 = df_f1.stack().mean()                 # macro f1 score
		print('Macro Precision:', macro_precision)
		print('Macro Recall:', macro_recall)
		print('Macro F1 score:', macro_f1)

		print('Micro Average')
		TP = 0    # true positive
		FP = 0    # false positive
		FN = 0    # false negative
		num_classes = len(classes)
		for i in range(num_classes):
			for j in range(num_classes):
				if i == j:
					TP += df[i][j]
				else:
					FP += df[i][j]
					FN += df[i][j]

		micro_precision = TP/(TP + FP)    # micro precision
		micro_recall = TP/(TP + FN)       # micro recall
		micro_f1 = harmonic_mean([micro_precision, micro_recall])   # micro f1 score
		print('Micro Precision:', micro_precision)
		print('Micro Recall:', micro_recall)
		print('Micro F1 score:', micro_f1)             
		roc(model.predict_proba(X), actual, classes)    # roc curve

def plot_graph_DT(GridSearch_results):       # plot of accuracy with max depth
	depth = GridSearch_results['parameters']
	max_depths = []                                        # max depth
	for d in depth:
		max_depths.append(d['max_depth'])
	val = list(GridSearch_results['val_accuracy'])         # validation accuracy
	train = list(GridSearch_results['train_accuracy'])     # training accuracy

	zipped_lists = zip(max_depths, val)                         # sort validation accuracies on the basis of sorted depth
	sorted_pairs = sorted(zipped_lists)
	tuples = zip(*sorted_pairs)
	graph_depth, val = [ list(tuple) for tuple in  tuples]

	zipped_lists = zip(max_depths, train)                       # sort training accuracies on the basis of sorted depth
	sorted_pairs = sorted(zipped_lists)
	tuples = zip(*sorted_pairs)
	graph_depth, train = [ list(tuple) for tuple in  tuples]

	plt.plot(graph_depth, val, label='val')                 # graphs
	plt.plot(graph_depth, train, label='train')
	plt.xlabel("Max Depth", fontsize=12)
	plt.ylabel("Accuracy", fontsize=12)
	plt.legend()
	plt.show()

def save_best_model(GridSearch_results, X_train, y_train):     # saving best model
	depth = GridSearch_results['parameters'][0]['max_depth']
	clf = DecisionTreeClassifier(max_depth=depth)
	clf.fit(X_train, y_train)
	filename = 'best_model.sav'
	pickle.dump(clf, open(filename, 'wb'))
	return filename

def predict_using_best_model(filename, X_test, y_test):       # loading best model and predicting on test data
	loaded_model = pickle.load(open(filename, 'rb'))
	y_pred = loaded_model.predict(X_test) 
	print('Test accuracy using best model', get_accuracy(y_pred, y_test))

# Dataset A

hf = h5py.File('part_A_train.h5', 'r')     # read .h5 file using h5py
X = np.array(hf.get('X'))      # get X
y = np.array(hf.get('Y'))      # get y
y = np.where(y==1)[1]
classes = np.unique(y)         # unique classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)    # train test split 80:20 with stratify

parameters = {'max_depth': range(1,30)}
results = GridSearch(parameters, X_train, y_train)      # grid search for decision tree
print(results)
plot_graph_DT(results)                                  # Decision Tree
filename = save_best_model(results, X_train, y_train)
predict_using_best_model(filename, X_test, y_test)

run_model_NGB(X_train, y_train)     # Gaussian Naive Bayes with k fold

clf = GaussianNB()                  # testing evaluation metrics for gaussian naive bayes
clf.fit(X_train, y_train)
y = clf.predict_proba(X_test)
pred = clf.predict(X_test)
evaulation_metric(pred, y_test, classes, clf, X_test)

clf = DecisionTreeClassifier(max_depth=10)     # testing evaluation metrics for decision tree
clf.fit(X_train, y_train)
y = clf.predict_proba(X_test)
pred = clf.predict(X_test)
evaulation_metric(pred, y_test, classes, clf, X_test)

# Dataset B

hf = h5py.File('part_B_train.h5', 'r')     # read .h5 file using h5py
X = np.array(hf.get('X'))      # get X
y = np.array(hf.get('Y'))      # get y
y = np.where(y==1)[1]
classes = np.unique(y)         # unique classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)   # train test split 80:20 with stratify

parameters = {'max_depth': range(1,20)}
results = GridSearch(parameters, X_train, y_train)       # grid search for decision tree
print(results)
plot_graph_DT(results)                                    # Decision Tree
best_model = save_best_model(results, X_train, y_train)
predict_using_best_model(best_model, X_test, y_test)

run_model_NGB(X_train, y_train)                          # Gaussian Naive Bayes with k fold

clf = GaussianNB()                                        # testing evaluation metrics for gaussian naive bayes
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
evaulation_metric(pred, y_test, classes, clf, X_test)

clf = DecisionTreeClassifier(max_depth=6)                 # testing evaluation metrics for decision tree
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
evaulation_metric(pred, y_test, classes, clf, X_test)