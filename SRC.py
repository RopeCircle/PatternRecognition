import time
from sklearn.linear_model import orthogonal_mp
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model


class SRC(object):
    def __init__(self, p, path, omp_nonzero):
        self.p = p
        self.path = path
        self.nonzero = omp_nonzero
        self.order = [1, 10, 11, 12, 13, 15, 16, 17, 18, 19,
                      2, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      29, 3, 30, 31, 32, 33, 34, 35, 36, 37,
                      38, 39, 4, 5, 6, 7, 8, 9]

    def read_data(self, file_path, mode, p):
        if mode == "train":
            with open(file_path + "train_"+str(p)+".txt", "r") as file:
                dataset = file.readlines()
                new = np.empty([len(dataset), 2016], dtype=float)
                tag = []
                for i, line in enumerate(dataset):
                    data = line.split(' ')
                    tag.append(int(data[0]))
                    del data[0]
                    for j, item in enumerate(data):
                        new[i][j] = float(item.split(':')[1])
            return new, tag
        elif mode == "test":
            with open(file_path + "test_"+str(p)+".txt", "r") as file:
                dataset2 = file.readlines()
                new2 = np.empty([len(dataset2), 2016], dtype=float)
                tag = []
                for i, line in enumerate(dataset2):
                    data = line.split(' ')
                    tag.append(int(data[0]))
                    del data[0]
                    for j, item2 in enumerate(data):
                        new2[i][j] = float(item2.split(':')[1])
            return new2, tag
        else:
            raise ValueError('Mode Error')

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def data_process(self, path):
        test_set, test_target = self.read_data(path, "test", self.p)
        test_set = np.array(test_set)
        test_target = np.array(test_target)

        train_set, train_target = self.read_data(path, "train", self.p)
        train_set = np.array(train_set)
        train_target = np.array(train_target)

        D = self.normalization(np.transpose(train_set))
        y = self.normalization(np.transpose(test_set))
        return D, y, test_target, train_target

    def fit_src(self):
        D, y, target, _ = self.data_process(self.path)
        true = 0
        all = y.shape[1]
        T1 = time.time()
        for j in range(all):
            result = orthogonal_mp(D, y[:, j], n_nonzero_coefs=self.nonzero)
            residual = []
            for i, t in enumerate(self.order):
                index = np.zeros((self.p*38))
                for k in range(self.p):
                    index[i * self.p + k] = 1
                xn = np.multiply(index, result)
                rebuilt = np.dot(D, xn)
                res = np.linalg.norm(y[:, j] - rebuilt)
                residual.append(res)
            if self.order[np.argmin(residual)] == target[j]:
                true += 1
        T2 = time.time()
        return true / all, (T2 - T1), (T2-T1)/all


'''
path = "D:/pr/yaleBExtData/"
y, target = read_data(path, "test", 7)
y = np.array(y)

dictionary, target2 = read_data(path, "train", 7)
dictionary = np.array(dictionary)
target2 = np.array(target2)
X = np.transpose(dictionary)
c = X.shape[0]
yt = np.transpose(y)
target2 = np.eye(len(target2))[target2.reshape(-1)].T * 0.01
X = normalization(X)
yt = normalization(yt)
print(statics(X, yt, target))

print(cs_omp(yt[:, 0], X))

X = np.vstack((X, target2))
ksvd = KSVD(200)
d, s = ksvd.fit(X)
np.save("dic", arr=d)

#d = np.load("dic.npy")
D = normalization(d[0:c, :])
C = normalization(d[c:, :]/0.01)
x = linear_model.orthogonal_mp(D, yt[:, 0:57])
print(np.argmax(np.dot(C, x), 0))

path = "D:/pr/yaleBExtData/"
method = SRC(7, path, 7)
print(method.fit())
'''