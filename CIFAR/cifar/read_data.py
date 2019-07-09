import pickle
import os
import numpy as np
from config import Config
import matplotlib.pyplot as plt

class Read_cifar10:

    def __init__(self):
        self.base_path = Config.data_path

    def _read_data(self, file_name):
        with open(file_name, 'rb') as f:
            dict = pickle.load(f, encoding = 'bytes')
        return dict[b'data'], dict[b'labels']

    def load_data(self, files):
        data, labels = [], []
        for file in files:
            data_n, label_n = self._read_data(file)
            label_n = np.array([[float(i == label) for i in range(10)] for label in label_n])

            data.append(data_n.reshape(10000, 3, 32, 32).transpose(0,2,3,1))
            labels.append(label_n)

        data = np.concatenate(np.array(data))
        labels = np.concatenate(np.array(labels))

        return data, labels

    def prepare_data(self):
        print('======loading data======')

        train_data_path = ['data_batch_%d.bin' % d for d in range(1, 6)]
        test_data_path = ['test_batch.bin']

        train_data_path = [os.path.join(self.base_path, file) for file in train_data_path]
        test_data_path = [os.path.join(self.base_path, file) for file in test_data_path]

        print('======loading traindata======')
        train_data, train_labels = self.load_data(train_data_path)
        test_data, test_labels = self.load_data(test_data_path)

        indices = np.random.permutation(train_data.shape[0])
        train_data = train_data[indices]
        train_labels = train_labels[indices]
        print("======Prepare Finished======")

        return train_data, train_labels, test_data, test_labels

    def data_preprocessing(self, x_train, x_test):

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
        x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
        x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

        x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
        x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
        x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

        return x_train, x_test

    def read_data(self):
        train_data_path = ['data_batch_%d.bin' % d for d in range(1, 6)]
        test_data_path = ['test_batch.bin']

        train_data_path = [os.path.join(self.base_path, file) for file in train_data_path]
        test_data_path = [os.path.join(self.base_path, file) for file in test_data_path]

        train_data, train_label = [], []
        test_data, test_label = [], []

        for path in train_data_path:
            data, labels = self._read_data(path)

            train_data.append(data.reshape(10000, 3, 32, 32).transpose(0,2,3,1))
            train_label.append(labels)

        self.train_data = np.concatenate(np.array(train_data))
        self.train_label = np.concatenate(np.array(train_label))
        self.train_label = np.array([[float(i == label) for i in range(10)] for label in self.train_label])

        for path in test_data_path:
            data, labels = self._read_data(path)
            test_data.append(data.reshape(10000, 3, 32, 32).transpose(0,2,3,1))
            test_label.append(labels)
        self.test_data = np.concatenate(np.array(test_data))
        self.test_label = np.concatenate(np.array(test_label))
        self.test_label = np.array([[float(i == label) for i in range(10)] for label in self.test_label])

        # train_data, train_label = self.load_data(train_data_path)
        # test_data, test_label = self.load_data(test_data_path)

        # train_data, train_label, test_data, test_label = self.prepare_data()

        #shuffle
        indices = np.random.permutation(self.train_data.shape[0])
        self.train_data = self.train_data[indices]
        self.train_label = self.train_label[indices]

        self.train_data, self.test_data = self.data_preprocessing(self.train_data, self.test_data)

        return self.train_data, self.train_label, self.test_data, self.test_label

if __name__ == '__main__':
    read = Read_cifar10()
    train_data, train_label, test_data, test_label = read.read_data()

    print(test_data.shape, test_label.shape)

