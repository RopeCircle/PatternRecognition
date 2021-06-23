from SRC import *
from DictionartLearning import *
from DataPreprocessing import data_preprocessing
from SVM import SVM


def run(mode, path, p):
    if mode == 'SRC':
        method = SRC(p, path, 7)
        acc, timeall, timeave = method.fit_src()
        print("SRC acc:%f ave_time:%f" % (acc*100, timeave))

    if mode == 'DL':
        method = D_KSVD(p, path, p, 0.01)
        acc, timeall, timeave = method.fit_ksvd()
        print("DctionartLearning acc:%f ave_time:%f" % (acc*100, timeave))

    if mode == 'SVM':
        method = SVM(p, path, '-c 4 -t 0 -q')
        acc, timeall, timeave = method.fit_svm()
        print("SVM acc:%f time:%f ave_time:%f" % (acc, timeall, timeave))

    if mode == 'DP':
        method = data_preprocessing(path, p)
        method.random_data()
        print("txt write done")


if __name__ == "__main__":
    path = "D:/pr/yaleBExtData/"
    p = 20

    for i in range(10):
        run('DP', path, p)
        run('SVM', path, p)
        run('SRC', path, p)
        run('DL', path, p)


