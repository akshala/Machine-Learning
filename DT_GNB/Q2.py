import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(42)

df = pd.read_csv('weight-height.csv')  # read csv file into pandas dataframe
df = df.dropna()

X = np.array(df['Height'])   # feature vector for height
y = np.array(df['Weight'])   # feature vector for weight

X = X.reshape((X.shape[0],-1))  # reshape array
y = y.reshape((y.shape[0],-1))  # reshape array

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   # train test split of 80:20

n = len(y_train)
choices = [i for i in range(n)]  # indices
test_length = len(y_test)
num_bootstrap_sample = 1000     # number of bootstrap samples
pred_list = np.zeros((num_bootstrap_sample, test_length))  # predicted list

for i in range(num_bootstrap_sample):
	indices = np.random.choice(choices, size=n)   # generate random indices to select for bootstrap samples
	bs_X = [X_train[j] for j in indices]          # X for bootstrap sample
	bs_y = [y_train[j] for j in indices]          # y for bootstrap sample
	model.fit(bs_X, bs_y)
	pred = model.predict(X_test)         
	pred = np.array(pred)
	pred = pred.reshape((pred.shape[0]))
	pred_list[i] = pred                           # add prediction to pred_list

avg_pred = np.mean(pred_list, axis=0)             # get average of predictions
bias = 0
for i in range(test_length):
	bias += abs(avg_pred[i] - y_test[i])          # calculating bias
bias /= test_length
print('Bias:', bias[0])

variance = 0
for i in range(num_bootstrap_sample):
	variance += np.square(pred_list[i] - avg_pred)   # calculating variance
variance /= (num_bootstrap_sample - 1)
variance = sum(variance)/test_length
print('Variance:', variance)

MSE = 0
for i in range(num_bootstrap_sample):
	MSE += mean_squared_error(pred_list[i], y_test)    # calculating MSE
MSE /= num_bootstrap_sample
print('MSE:', MSE)

print('MSE − Bias2 − Variance:', MSE - (bias**2) - variance)    # calculating noise




