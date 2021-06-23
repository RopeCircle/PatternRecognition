import time

import numpy as np
from sklearn.linear_model import orthogonal_mp
from SRC import SRC
import os


class KSVD(object):
    def __init__(self, n_components, max_iter=30, tol=1e-6,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)
            print(i)

        self.sparsecode = orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


class D_KSVD(SRC):
    def __init__(self, p, path, omp_nonzero, u, iter=30):
        super(D_KSVD, self).__init__(p, path, omp_nonzero)
        self.norm = u
        if p == 13:
            self.size = 400
        elif p == 20:
            self.size = 600
        else:
            self.size = 200
        self.iter = iter

    def train(self, trainset, traintarget):
        file = os.listdir("./")
        file_to_load = "dic_"+str(self.p)+".npy"
        if file_to_load in file:
            return np.load(file_to_load)
        else:
            target = np.eye(len(traintarget))[traintarget.reshape(-1)].T * self.norm
            X = np.vstack((trainset, target))
            ksvd = KSVD(self.size, max_iter=self.iter)
            T1 = time.time()
            d, s = ksvd.fit(X)
            T2 = time.time()
            print("training time:", T2-T1)
            np.save("dic_"+str(self.p), arr=d)
            return d

    def fit_ksvd(self):
        trainset, testset, testtarget, traintarget = self.data_process(self.path)
        c = trainset.shape[0]
        d = self.train(trainset, traintarget)
        D = self.normalization(d[0:c, :])
        C = self.normalization(d[c:, :]/self.norm)
        T1 = time.time()
        x = orthogonal_mp(D, testset)
        result = np.argmax(np.dot(C, x), 0)
        true = 0
        for i in range(len(testtarget)):
            if result[i] == testtarget[i]:
                true += 1
        T2 = time.time()
        return true/len(testtarget), T2-T1, (T2-T1)/len(testtarget)


'''
patht = "D:/pr/yaleBExtData/"
method = D_KSVD(7, patht, 7, 0.01)
print(method.fit_ksvd())
'''