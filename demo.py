import random
import itertools
import numpy as np
from collections import defaultdict
from metric_learn import LMNN,NCA,ITML_Supervised

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.datasets import load_iris,load_digits,load_breast_cancer
from sklearn import neighbors

import torch
from torch.autograd import Variable
import torch.nn.functional as F



def make_metric_data(X, y):
	new_X = []
	new_y = []
	positive_rate = 0.67
	negative_rate = 0.33
	for p1,p2 in itertools.product(zip(X,y),repeat=2):
		feature1, label1 = p1
		feature2, label2 = p2
		if label1==label2 and random.random()<positive_rate:
			new_X.append(np.concatenate((feature1,feature2)))
			new_y.append(1)
			positive_rate *= 0.8
			negative_rate = 1 - positive_rate
		elif label1!=label2 and random.random()<negative_rate:
			new_X.append(np.concatenate((feature1,feature2)))
			new_y.append(10)
			negative_rate *= 0.8
			positive_rate = 1 - negative_rate
	# print (sum(new_y)/len(new_y),len(new_y),len(y))
	return np.array(new_X), np.array(new_y)



class Simple_Version:
	def __init__(self,n_feature, n_hidden=12, net_num=10):
		net_gene = lambda :torch.nn.Sequential(
				torch.nn.Linear(n_feature, n_hidden),
				torch.nn.ReLU(),
				torch.nn.Linear(n_hidden, n_hidden),
				torch.nn.ReLU(),
				torch.nn.Linear(n_hidden, 1),
				)
		self.nets = [net_gene() for i in range(net_num)]
		self.optimizers = [torch.optim.RMSprop(net.parameters(), lr=0.02, alpha=0.9) for net in self.nets]
		self.loss_func  = torch.nn.MSELoss()

	def fit(self, X, y):
		for net,optimizer in zip(self.nets,self.optimizers):
			_X, _y = make_metric_data(X, y)
			_X = torch.FloatTensor(_X)
			_y = torch.FloatTensor(_y)
			_X, _y = Variable(_X),Variable(_y)
			print ('\n-->')
			for t in range(2000):
				if t%1000 == 0:
					print (t,end=" ")
				prediction = net(_X)
				loss = self.loss_func(prediction, _y)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

	def distance(self, x1, x2):
		d = np.zeros(1)
		for net in self.nets:
			X = np.concatenate((x1,x2))
			X = torch.FloatTensor(X)
			X = Variable(X)
			d += net(X).data.numpy()
		return d

def load_data():
	iris_data = load_digits () #load_iris()
	X = iris_data['data']
	y = iris_data['target']
	new_X = []
	new_y = []
	dic = defaultdict(int)
	for img,label in zip(X,y):
		if dic[label]>=10:
			continue
		new_X.append(img)
		new_y.append(label)
		dic[label]+=1
	print (new_y,len(new_y))
	return np.array(new_X),np.array(new_y)

def knn_method(X, y, splits=6):
	skf = StratifiedKFold(n_splits=splits)
	accuracy = 0
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		knn = neighbors.KNeighborsClassifier()
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		a = sum(y_pred == y_test)/len(y_test)
		print (a)
		print (y_pred)
		print (y_test)
		accuracy += a
	return (accuracy/splits)


def lmnn_method(X, y, splits=6):
	skf = StratifiedKFold(n_splits=splits)
	accuracy = 0
	lmnn = LMNN(k=3, learn_rate=1e-6)
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		lmnn.fit(X_train, y_train)
		X_train = lmnn.transform(X_train)
		X_test  = lmnn.transform(X_test)
		
		knn = neighbors.KNeighborsClassifier()
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		a = sum(y_pred == y_test)/len(y_test)

		print ('\t>>',a)
		accuracy += a
	return (accuracy/splits)


def our_method(X, y, splits=6):
	skf = StratifiedKFold(n_splits=splits)
	accuracy = 0
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		sv = Simple_Version(64*2)	
		sv.fit(X_train, y_train)

		print ('度量学习完毕')
		knn = neighbors.KNeighborsClassifier(metric=sv.distance)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)

		a = sum(y_pred == y_test)/len(y_test)
		print ('\t>>',a)
		accuracy += a
	return (accuracy/splits)

if __name__ == "__main__":
	X,y = load_data()
	# fs = (knn_method,lmnn_method,our_method)
	fs = (knn_method,)
	for f in fs:
		accuracy=f(X,y,6)
		print (accuracy)