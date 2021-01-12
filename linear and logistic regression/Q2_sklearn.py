from sklearn.linear_model import LogisticRegression
from scratch import MyPreProcessor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

preProcessor = MyPreProcessor()
X, y = preProcessor.pre_process(2)

# train test 80:20 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

clf = LogisticRegression()   # normal logistic regression
clf.fit(X_train, y_train) 
y_predict = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predict, normalize=True, sample_weight=None)   # accuracy
print('accuracy using sklearn logistic regression', accuracy) 

sgd = SGDClassifier(loss='log', max_iter=10000, learning_rate='constant',eta0=0.01)   # SGD logistic regression
sgd.fit(X_train, y_train)

y_predict = sgd.predict(X_test)
accuracy = accuracy_score(y_test, y_predict, normalize=True, sample_weight=None)   # test accuracy
print('test accuracy using sklearn SGD classifier', accuracy)

y_predict = sgd.predict(X_train)
accuracy = accuracy_score(y_train, y_predict, normalize=True, sample_weight=None)   # ttrain accuracy
print('train accuracy using sklearn SGD classifier', accuracy)

