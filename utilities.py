'''
utilities for importing data and plotting
'''
from pathlib import Path
import subprocess
import struct
import numpy as np
import matplotlib.pyplot as plt


def idx2numpy(fname):
    '''
    function to convert idx files to numpy array
    '''
    with open(fname, 'rb') as data:
        # get no. of dimensions from the header
        _, _, dims = struct.unpack('>HBB', data.read(4))
        # find shape as per no. of dimensions
        shape = list(struct.unpack('>I', data.read(4))[0] for _ in range(dims))
        # read everything else in the file and reshape as required
        return np.frombuffer(data.read(), dtype=np.uint8).reshape(shape)


class MNIST:
    '''
    class for handling mnist data
    '''
    def __init__(self):
        '''
        constructor
        initializes the variables
        downloads the mnist dataset if required
        '''
        self.epochs_completed = 0
        self.train_idx = 0
        # links for data
        download_links = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                          'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                          'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                          'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
        # download data if unavailable
        if not Path('./MNIST').exists():
            subprocess.run(['mkdir', 'MNIST'])
            for link in download_links:
                name = link.split('/')[-1]
                subprocess.run(['wget', link, '-P', './MNIST'])
                subprocess.run(['gunzip', './MNIST/'+name])
        # load everything into memory as the files are small
        (self.train_i,
         self.train_l,
         self.test_i,
         self.test_l) = (idx2numpy('./MNIST/'+link.split('/')[-1].split('.')[0])
                         for link in download_links)
        # reshape images and adjust range of values
        self.train_i = self.train_i.reshape((-1, 784))/255
        self.test_i = self.test_i.reshape((-1, 784))/255
        # make labels one hot
        buff = np.zeros((self.train_l.shape[0], 10))
        buff[np.arange(self.train_l.shape[0]), self.train_l] = 1
        self.train_l = np.copy(buff)
        buff = np.zeros((self.test_l.shape[0], 10))
        buff[np.arange(self.test_l.shape[0]), self.test_l] = 1
        self.test_l = np.copy(buff)

    def next_batch(self, b_size):
        '''
        returns train data of given batch size
        '''
        # get the indices
        indices = np.arange(self.train_idx, self.train_idx+b_size) % self.train_i.shape[0]

        # shift the starting location
        self.train_idx += b_size
        if self.train_idx > self.train_i.shape[0]:
            self.epochs_completed += 1
            self.train_idx %= self.train_i.shape[0]

        return (self.train_i[indices, :],
                self.train_l[indices, :])

    def test_batch(self):
        '''
        returns the entire test data
        '''
        return np.copy(self.test_i), np.copy(self.test_l)


def summarize_result(accuracies):
    '''
    method to plot learning curve - test accuracy
    '''
    print("Achieved Prediction Accuracy: ", accuracies[-1], "%")
    plt.plot(np.arange(1, len(accuracies)+1)/3, accuracies)
    plt.ylim(0.85, 1.00)
    plt.title("Learning Curve")
    plt.ylabel('Test Accuracy %')
    plt.xlabel('Epochs')
    plt.show()
