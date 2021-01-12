# -*- coding: utf-8 -*-
"""ML_Q4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nK4cX7gC01Hk8d2in9qOBUW7sEaNZ9UA
"""

# from google.colab import drive
# drive.mount('/content/drive')

# cd /content/drive/MyDrive/ML_HW3

# !pip install scikit-plot

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import random
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, plot_roc_curve
import scikitplot as skplt
import PIL

"""Setting seed for random values"""

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

"""Read data"""

train_object = pd.read_pickle('train_CIFAR.pickle')
test_object = pd.read_pickle('test_CIFAR.pickle')

y_train = train_object['Y']
X_train = train_object['X']
y_test = test_object['Y']
X_test = test_object['X']

"""Load alexnet model"""

alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
alexnet.eval()

"""Defining transforms for images"""

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""Show sample images"""

for image in X_train:
  image = image.reshape((3, 32,32))
  image = image.transpose(1,2,0)
  plt.imshow(image)

"""Feature extraction using alexnet"""

extracted_features_train = []
for image in X_train:
  image = image.reshape((3, 32,32))   # reshape to 3,32,32 - features represented this way
  image = image.transpose(1,2,0)      # change shape to 32, 32, 3
  image = transforms.ToPILImage()(image)  # convert to PIL image
  input_tensor = preprocess(image)        # apply transforms
  input_batch = input_tensor.unsqueeze(0)
  with torch.no_grad():
    output = alexnet(input_batch)     # fc8 layer output of alexnet - its the last layer
  output = output.reshape((1000))
  extracted_features_train.append(np.array(output))

extracted_features_train = np.array(extracted_features_train)  # train feature array

extracted_features_test = []
for image in X_test:
  image = image.reshape((3, 32,32))   # reshape to 3,32,32 - features represented this way
  image = image.transpose(1,2,0)      # change shape to 32, 32, 3
  image = transforms.ToPILImage()(image)  # convert to PIL image
  input_tensor = preprocess(image)         # apply transforms
  input_batch = input_tensor.unsqueeze(0)
  with torch.no_grad():
    output = alexnet(input_batch)      # fc8 layer output of alexnet - its the last layer
  output = output.reshape((1000))
  extracted_features_test.append(np.array(output))

extracted_features_test = np.array(extracted_features_test)  # test feature array

class Dataset(torch.utils.data.Dataset):

	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		return self.X[index], self.Y[index]

class MLP(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
            
    self.input_fc = nn.Linear(input_dim, 512)
    self.hidden_fc = nn.Linear(512, 256)
    self.output_fc = nn.Linear(256, output_dim)
        
  def forward(self, x):
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)

    hidden1 = F.tanh(self.input_fc(x))
    hidden2 = F.tanh(self.hidden_fc(hidden1))
    y_pred = self.output_fc(hidden2)
    return y_pred, hidden2

def calculate_accuracy(y_pred, y):
  """
        Calculating accuracy

        Parameters
        ----------
        y_pred : 2-dimensional numpy array of shape (n_samples, n_classes) which is the predicted value
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which is the predicted value

        Returns
        -------
        acc : accuracy
        """
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, optimizer, criterion, iterator, device):
    """
        Training model

        Parameters
        ----------
        model : neural network model to be used for training
        optimizer : optimizer used
        criterion : loss function used
        iterator : Dataset type object which contains train data
        device : for using GPU

        Returns
        -------
        accuracy
        """
  epoch_acc = 0
  model.train()
  for (x, y) in iterator:
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()     
    y_pred, _ = model(x)       
    acc = calculate_accuracy(y_pred, y)
    loss = criterion(y_pred, y)
    loss.backward() 
    optimizer.step()
    epoch_acc += acc.item()
  return epoch_acc / len(iterator)

def evaluate(model, iterator, device):   
"""
        Evaluating model

        Parameters
        ----------
        model : neural network model to be used for training
        iterator : Dataset type object which contains train data
        device : for using GPU

        Returns
        -------
        accuracy
        """ 
  epoch_acc = 0
  model.eval()  
  with torch.no_grad():  
    predicted = np.empty((0,2), float)
    for (x, y) in iterator:
      x = x.to(device)
      y = y.to(device)
      y_pred, _ = model(x)
      acc = calculate_accuracy(y_pred, y)
      epoch_acc += acc.item()
      prob = F.softmax(y_pred)
      prob = prob.cpu().data.numpy()
      predicted = np.append(predicted, prob, axis=0)
  return predicted, epoch_acc / len(iterator)

train_iterator = data.DataLoader(Dataset(extracted_features_train, y_train), shuffle = True, batch_size = 64)   # train
test_iterator = data.DataLoader(Dataset(extracted_features_test, y_test), batch_size = 64)                      # test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # for GPU use

model = MLP(1000, 2)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model.to(device)                      # for GPU use
criterion = criterion.to(device)      # for GPU use

for epoch in range(100):  
	train_acc = train(model, optimizer, criterion, train_iterator, device)
y_pred, test_acc = evaluate(model, test_iterator, device)

print('Train accuracy', train_acc)
print('Test accuracy', test_acc)

score = y_pred
y_pred = y_pred.argmax(axis=1)
skplt.metrics.plot_roc_curve(y_test, score)   # plotting roc curve
print(confusion_matrix(y_test, y_pred))       # confusion matrix
