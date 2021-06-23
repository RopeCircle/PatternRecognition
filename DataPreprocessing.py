from PIL import Image
import random
import os
import numpy as np


class data_preprocessing(object):
    def __init__(self, path, p):
        self.path = path
        self.p = p

    def fomat_trans(self):
        '''
            :exception
                将所有图片数据转换成libsvm格式，保存为txt文件
        '''
        fold = os.listdir(self.path)
        for x in fold:
            file_path = self.path + x + '/'
            imgs = []
            for y in os.listdir(file_path):
                if not y.endswith('Ambient.pgm'):
                    imgs.append(y)
            data = self.to_vector(imgs, file_path)
            self.write_txt(data, int(x[-2:]))

    def to_vector(self, dataset, path):
        '''
            :exception
                读取数据集文件并转换成向量
            :parameter
                dataset - 数据集
            :return
                向量
        '''
        vector = []
        for x in dataset:
            im = Image.open(path + x)  # 读取文件
            im = im.resize((42, 48), Image.ANTIALIAS)
            im = np.array(im)
            vector.append(im)
        return np.array(vector)

    def write_txt(self, dataset, label):
        with open(self.path + str(label) + '.txt', 'a', encoding='utf-8', newline="") as file:
            for x in dataset:
                x = x.flatten()
                file.write(str(label))
                for i, y in enumerate(x):
                    file.write(' ' + str(i) + ':' + str(y))
                file.write('\r\n')

    def random_data(self):
        '''
            :exception
                每个人脸随机挑选num个数据作为训练集，其余作为验证集，分别形成txt文件
            :parameter
                path - 根目录
                num - 训练集大小
        '''

        filelist = os.listdir(self.path)
        train_dataset = []
        test_dataset = []
        for name in filelist:
            if name.endswith('.txt') and not name.startswith('test') and not name.startswith('train'):
                with open(self.path + name, 'r') as file:
                    dataset = file.readlines()
                randomdata = random.sample(dataset, self.p)
                train_dataset = train_dataset + randomdata
                for item in randomdata:
                    dataset.remove(item)
                test_dataset = test_dataset + dataset

        with open(self.path + 'train_' + str(self.p) + '.txt', 'w', encoding='utf-8', newline="") as TrainDataset:
            for item in train_dataset:
                TrainDataset.write(item)

        with open(self.path + 'test_' + str(self.p) + '.txt', 'w', encoding='utf-8', newline="") as TestDataset:
            for item in test_dataset:
                TestDataset.write(item)


'''
if __name__ == "__main__":
    path = "D:/pr/yaleBExtData/"
    fomat_trans(path)
    random_data(path, 7)
'''





