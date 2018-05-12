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


def knn(X, y, splits=5):
    num, xlen, ylen = X.shape
    X = X.reshape((num, xlen * ylen))
    accuracy = 0
    mskf = my_SKF(X, y, splits)

    cnt = 0
    for X_train, y_train, X_test, y_test in mskf:
        knn = neighbors.KNeighborsClassifier()
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracy += acc
        cnt += 1
        print('Split:{}, Acc:{:.4f}'.format(cnt, acc))
    return accuracy / splits


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


def knn_pred(net, X_train, Y_train, x, batch_num=2, k=5):
    batch_size = int(X_train.shape[0] / batch_num)
    batch_x1 = np.array([x] * batch_size)
    candidates = []
    for i in range(batch_num):
        batch_x2 = X_train[i * batch_size:(i + 1) * batch_size]
        input1 = torch.Tensor(batch_x1).unsqueeze(1).cuda()
        input2 = torch.Tensor(batch_x2).unsqueeze(1).cuda()
        batch_distance = net.distance(input1, input2).squeeze()
        topk, idxs = torch.topk(batch_distance, k, largest=False)
        idxs = idxs + i * batch_size
        distance_idx_pairs = list(zip(topk.tolist(), idxs.tolist()))
        candidates += distance_idx_pairs
    candidates.sort()
    topk_candidates = candidates[:k]
    topk_idxs = [p[1] for p in topk_candidates]
    topk_labels = Y_train[topk_idxs].tolist()
    label_count = dict((x, topk_labels.count(x)) for x in set(topk_labels))
    max_count = max(label_count.values())
    for label in label_count:
        count = label_count[label]
        if count == max_count:
            return label


def transform(X, Y):
    new_X = [[] for i in range(max(Y) + 1)]
    for x, y in zip(X, Y):
        new_X[y].append(x)
    return new_X


def mlnet_method(X, Y, splits=5):
    accuracy, cnt = 0, 0
    mskf = my_SKF(X, Y, splits)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for X_train, Y_train, X_test, Y_test in mskf:
        new_X = transform(X_train, Y_train)
        dataset = ImageDataset(new_X)
        dataloader = DataLoader(
            dataset, batch_size=128, shuffle=True, num_workers=4)
        net = SiameseNetwork().to(device)
        net.fit(dataloader)

        net.eval()
        hit = 0
        for i, x in enumerate(X_test):
            y_pred = knn_pred(net, X_train, Y_train, x, batch_num=1)
            y_real = Y_test[i]
            if y_pred == y_real:
                hit += 1
        acc = hit / X_test.shape[0]
        accuracy += acc
        cnt += 1
        print('Split:{}, Acc:{:.4f}'.format(cnt, acc))
    return accuracy / splits


if __name__ == "__main__":
    X, Y = load_data()
    splits = 10
    
    # run original knn on 10-flod cross validation sets
    print('Vanilla KNN:')
    accuracy = knn(X, Y, splits=splits)
    print('mean Acc:{:.4f}'.format(accuracy))
    
    # run siamesenetwork-based knn on 10-flod cross validation sets
    print('SiameseNetwork-based KNN:')
    accuracy = mlnet_method(X, Y, splits=splits)
    print('mean Acc:{:.4f}'.format(accuracy))