from libsvm.svmutil import *
from time import time
import numpy as np


class SVM(object):
    def __init__(self, p, path, parameter):
        self.path = path
        self.p = p
        self.parameter = parameter

    def train(self):
        y, x = svm_read_problem(self.path+'train_'+str(self.p)+'.txt')
        T1 = time()
        model = svm_train(y, x, self.parameter)
        T2 = time()
        return model, T2-T1

    def fit_svm(self):
        model, train_time = self.train()
        if '-v' in self.parameter:
            return 0, 0, 0
        y, x = svm_read_problem(self.path+'test_'+str(self.p)+'.txt')
        T1 = time()
        labels, result, _ = svm_predict(y, x, model)
        T2 = time()
        return result[0], train_time, (T2-T1)/len(labels)

