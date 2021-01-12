import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

hf = h5py.File('part_A_train.h5', 'r')     # read .h5 file using h5py
X = np.array(hf.get('X'))      # get X
y = np.array(hf.get('Y'))      # get y
y = np.where(y==1)[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)    # stratified sampling 80:20 train test split

df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)

train_column_sum = df_y_train.sum(axis=0)   # number of occurences of each class in train
test_column_sum = df_y_test.sum(axis=0)     # number of occurences of each class in test

percentage_train = train_column_sum/train_column_sum.sum(axis=0)    # percentage of occurences of each class in train
percentage_test = test_column_sum/test_column_sum.sum(axis=0)       # percentage of occurences of each class in test

pca = PCA(n_components=200, random_state=42)    # random_state set to 42 so that results can be reproduced
pca.fit(X_train)        # fit PCA on X_train
pca_train = pca.transform(X_train)     # transform X_train
# print(pca_train.shape)
pca_test = pca.transform(X_test)       # transform X_test

model = LogisticRegression(max_iter=5000, random_state=0)
model.fit(pca_train, y_train)     # fit model on train data
y_pred = model.predict(pca_test)  # predict using test data
print('PCA')
print('Train accuracy', model.score(pca_train, y_train))
print('Test accuracy', accuracy_score(y_test, y_pred))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_train)         # fit t-SNE on train data

features = ['num' + str(i) for i in range(pca_train.shape[1])]
df = pd.DataFrame(pca_train, columns=features)          # new dataframe with X and y values
df['y'] = y_train

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


svd = TruncatedSVD(n_components=200, random_state=42)    # random_state set to 42 so that results can be reproduced
svd.fit(X_train)        # fit SVD on X_train
svd_train = svd.transform(X_train)      # transform X_train
svd_test = svd.transform(X_test)        # transform X_test

model = LogisticRegression(max_iter=5000, random_state=0)
model.fit(svd_train, y_train)           # fit model on train data
y_pred = model.predict(svd_test)         # predict using test data
print('SVD')
print('Train accuracy', model.score(svd_train, y_train))
print('Test accuracy', accuracy_score(y_test, y_pred))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(svd_train)       # fit t-SNE on train data

features = ['num' + str(i) for i in range(svd_train.shape[1])]
df = pd.DataFrame(svd_train, columns=features)       # new dataframe with X and y values
df['y'] = y_train

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(8,8))
sns.scatterplot(                      # plotting t-SNE graph
	x='tsne-2d-one', y='tsne-2d-two',
	hue='y',
	palette=sns.color_palette('hls', 10),
	data=df,
	legend='full',
	alpha=0.3
)
plt.show()


