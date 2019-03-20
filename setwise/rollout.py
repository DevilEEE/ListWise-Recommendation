# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:37:48 2019

@author: v_fdwang
"""
import os
import random
import math
import copy

import tqdm

import numpy as np

import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, num, evaluator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            evaluator : evaluator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data)
                pred = evaluator(samples)
                pred = pred.cpu().data[:,0].numpy()
                #print(pred)
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = evaluator(x)
            pred = pred.cpu().data[:, 0].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            #if name.startswith('emb'): # for pretrained word_vec!
                #param.data = dic[name]
            if name.startswith('emb'):
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]



class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = t.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(t.ByteTensor)
        one_hot = V(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = t.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -t.sum(loss)
        return loss