import numpy as np

from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold

# metric_learn 中的 lmnn 存在一些错误，需要安装 shogun
# pylmnn 会导致 内存不足
# from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
# from metric_learn import LMNN,Covariance
from torch.utils.data import Dataset, DataLoader
from load_data import load_all_data, matrix_reshape
from model import SiameseNetwork, ImageDataset
import torch


def load_data():
    all_data = load_all_data()
    X = []
    y = []
    for people, faces in enumerate(all_data):
        for face in faces:
            X.append(face)
            y.append(people)
    return np.array(X), np.array(y)


def X_reshape(X, rate=0.5):
    # rate * rate
    new_X = []
    for x in X:
        new_x = matrix_reshape(x, rate)
        new_X.append(new_x)
    return np.array(new_X)


class my_SKF:
    def __init__(self, X, y, splits):
        self.X = X
        self.y = y
        self.skf = StratifiedKFold(n_splits=splits)
        self.skf = self.skf.split(X, y)

    def __next__(self):
        train_index, test_index = self.skf.__next__()
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        return X_train, y_train, X_test, y_test

    def __iter__(self):
        return self


def knn(X, y, splits=6):
    num, xlen, ylen = X.shape
    X = X.reshape((num, xlen * ylen))
    accuracy = 0
    mskf = my_SKF(X, y, splits)

    for X_train, y_train, X_test, y_test in mskf:
        knn = neighbors.KNeighborsClassifier()
        knn.fit(X_train, y_train)
        a = knn.score(X_test, y_test)
        accuracy += a
    return (accuracy / splits)


def lmnn(X, y, splits=6, rate=0.1):
    # rate 进行图片放缩
    X = X_reshape(X, 0.3)
    num, xlen, ylen = X.shape
    print(X.shape)
    X = X.reshape((num, xlen * ylen)).astype(float)

    accuracy = 0
    mskf = my_SKF(X, y, splits)
    lmnn = LMNN(k=1, learn_rate=1e-6, use_pca=False, max_iter=1000)

    for X_train, y_train, X_test, y_test in mskf:
        lmnn.fit(X_train, y_train)
        X_train = lmnn.transform(X_train)
        X_test = lmnn.transform(X_test)

        knn = neighbors.KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        a = sum(y_pred == y_test) / len(y_test)
        print('\t', a)
        accuracy += a
    return (accuracy / splits)


def transform(X, Y):
    new_X = [[]] * (max(Y) + 1)
    for x, y in zip(X, Y):
        new_X[y].append(x)
    return new_X


def mlnet_method(X, y, splits=6):
    accuracy = 0
    mskf = my_SKF(X, y, splits)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for X_train, y_train, X_test, y_test in mskf:
        new_X = transform(X_train, y_train)
        dataset = ImageDataset(new_X)
        dataloader = DataLoader(
            dataset, batch_size=128, shuffle=True, num_workers=4)
        print('data loaded...')
        mlnet = SiameseNetwork().to(device)
        print('model loaded...')
        print('start training...')
        mlnet.fit(dataloader)

        knn = neighbors.KNeighborsClassifier(metric=mlnet.distance)
        knn.fit(X_train, y_train)
        a = knn.score(X_test, y_test)
        accuracy += a
    return (accuracy / splits)


if __name__ == "__main__":
    X, y = load_data()
    accuracy = mlnet_method(X, y, 5)
    print(accuracy)
