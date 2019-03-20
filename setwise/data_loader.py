# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:39:07 2019

@author: v_fdwang
"""

#train_path = 'F:/data/train_data.txt'
import json
import random
import torch as t
import numpy as np
import math

class EvaDataIter(object):
    def __init__(self, padding_length=10, batch_size=128, split=0.8, data_path='F:/nothing.txt'):
        #
        super(EvaDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.padding_length = padding_length
        
        self.data = []
        
        self.train_idx = 0
        self.test_idx = 0
        
        
        #
        self.load_data(data_path)
        self.train_test_split(split)
        
    def __next__(self):
        return self.next()
    
    def next(self):
        if self.train_idx >= len(self.train_data):
            self.train_idx = 0
        if self.test_idx >= len(self.test_data):
            self.test_idx = 0
        train_batch = np.array(self.train_data[self.train_idx:self.train_idx+self.batch_size])
        test_batch = np.array(self.test_data)
        train_x = t.LongTensor(train_batch[:, :-1])
        train_y = t.Tensor(train_batch[:, -1:])
        test_x = t.LongTensor(test_batch[:, :-1])
        test_y = t.Tensor(test_batch[:, -1:])
        self.train_idx += self.batch_size
        #self.test_idx += self.batch_size
        return train_x, train_y, test_x, test_y
    
    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                feature = line['feature'].strip().split(',')
                label = float(line['label'].strip())
                sample = [int(w)+1 for w in feature]
                while len(sample) < self.padding_length:
                    sample.append(0)
                if len(sample) > 10:
                    continue
                sample.append(label)
                self.data.append(sample)
            
    def train_test_split(self, split):
        data_num = len(self.data)
        train_num = int(split * data_num)
        random.shuffle(self.data)
        self.train_data = self.data[:train_num]
        self.test_data = self.data[train_num:]
        
        
#data_iter = DataIter()    
class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data_path='F:/nothing.txt', batch_size=1000):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.padding_length = 10
        self.data = []
        
        self.data_lis = self.read_file(data_path)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = t.LongTensor(np.asarray(d, dtype='int64'))
        data = t.cat([t.zeros(self.batch_size, 1).long(), d], dim=1)
        target = t.cat([d, t.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                feature = line['feature'].strip().split(',')
                label = float(line['label'].strip())
                sample = [int(w)+1 for w in feature]
                while len(sample) < self.padding_length:
                    sample.append(0)
                if len(sample) > 10:
                    continue
                sample.append(label)
                self.data.append(sample)
        return self.data
      
            
            
            

                