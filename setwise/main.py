# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:25:37 2019

@author: v_fdwang
"""
import os
import warnings
import math
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.autograd import Variable as V
from torch import nn
from torch import optim

from data_prepare import DataPre
from data_loader import EvaDataIter, GenDataIter
from model import Model, Generator
from rollout import Rollout, GANLoss
# ignore warnings
warnings.filterwarnings("ignore")
# parameter config
config = {
        "dic_size": 200,
        "emb_size": 300,
        "max_push_len": 10,
        "lin1_size": 300,
        "out_size": 1,
        "use_cuda": False,
        "num_emb": 176,
        "hidden_dim": 300,
        "seed": 1234
        }
random.seed(config["seed"])
np.random.seed(config["seed"])

# train EVAluator with MSEloss!
RAW_FILE_PATH = 'F:/data/test_7_20190311'
PRE_EVA_PATH = 'F:/data/model.pth'
PRE_GEN_PATH = 'F:/data/generator.pth'
TRAIN_NEW_EVA = False

print("EVA period+++++++++++EVA period+++++++++++++++++EVA period+++++++++++++++++++++++")
if TRAIN_NEW_EVA:
    print("TRAIN MODE OPEN FOR EVA! ++++++++++++++++++++++++++++++++++++++++++++++++++++")
    data_pre = DataPre(RAW_FILE_PATH, reset_dic=True)
    
    TRAIN_BATCHES = 10000

    evaluator = Model(config)
    data_iter = EvaDataIter(batch_size=1000, split=0.95, data_path=data_pre.get_data_path())
    optimizer = optim.Adagrad(evaluator.parameters())
    criterion = nn.MSELoss()

    plot_train_x = []
    plot_train_y = []
    plot_test_x = []
    plot_test_y = []
    
    print("DataSize: ",len(data_iter.data))

    print("Training! ----------------------------------------------------------")
    for epoch in range(TRAIN_BATCHES):
        train_x, train_y, test_x, test_y = next(data_iter)
        train_x, train_y = V(train_x), V(train_y, requires_grad=False)
        test_x, test_y = V(test_x), V(test_y, requires_grad=False)
        # forward
        y = evaluator(train_x)
        optimizer.zero_grad()
        loss = criterion(y, train_y)
        if epoch % 1 == 0:
            print("Batch: ", epoch, "   Train Loss: ", loss.data.numpy())
        loss.backward()
        optimizer.step()
    
        plot_train_x.append(epoch)
        plot_train_y.append(loss.data.numpy())
    
        if epoch % 100 == 99:
            print("Testing! ***************************************************")
            ty = evaluator(test_x)
            tloss = criterion(ty, test_y)
            print("TIME: ", (epoch+1)/100, "   Test Loss: ", tloss.data.numpy())
            print('\n##########################################################')
            
            plot_test_x.append((epoch+1)/100)
            plot_test_y.append(tloss.data.numpy())
            
            a = ty.data.numpy()[:,0].tolist()
            b = test_y.data.numpy()[:,0].tolist()
            c = test_x.data.numpy()[:,:10].tolist()
            m = list(zip(a, b, c))
            o = sorted(m, key = lambda x: -x[0])
            result = []
            for item in o:
                result.append(item[1])
            avg_list = []
            for i in range(5):
                avg_list.append(sum(result[:int((i+1)/5*len(result))]) / int((i+1)/5*len(result)))
            print(avg_list)
            base = avg_list[-1]
            zengfu_list = [w/base -1 for w in avg_list]
            print(zengfu_list)
            # check diversity
            print("top 20!")
            for i in range(20):
                print(o[i][2])
            print("************************************************************")
        
    
    plt.plot(plot_train_x[20:], plot_train_y[20:])
    plt.plot(plot_test_x, plot_test_y)
    
    t.save(evaluator, PRE_EVA_PATH)
    
        
else:
    print("TEST MODE OPEN FOR EVA! ++++++++++++++++++++++++++++++++++++++++++++++++++++")
    if not os.path.isfile(PRE_EVA_PATH):
        raise ValueError("Model not exists, make sure EVA model trained first and check your EVA model path!")
    
    data_pre = DataPre(RAW_FILE_PATH)
    evaluator = t.load(PRE_EVA_PATH)
    data_iter = EvaDataIter(batch_size=1000, split=0.95, data_path=data_pre.get_data_path())
    _, _, test_x, test_y = next(data_iter)
    test_x, test_y = V(test_x), V(test_y, requires_grad=False)
    ty = evaluator(test_x)
    
    a = ty.data.numpy()[:,0].tolist()
    b = test_y.data.numpy()[:,0].tolist()
    m = list(zip(a, b))
    o = sorted(m, key = lambda x: -x[0])
    result = []
    for item in o:
        result.append(item[1])
    avg_list = []
    for i in range(5):
        avg_list.append(sum(result[:int((i+1)/5*len(result))]) / int((i+1)/5*len(result)))
    print(avg_list)
    base = avg_list[-1]
    zengfu_list = [w/base -1 for w in avg_list]
    print(zengfu_list)


# pre-train for generator!
print("Generator period++++++++++++++++++Generator period+++++++++++++++++++++Generator period++++++")
PRE_EPOCH_NUM = 1
OUT_FILE = "F:/gen_samples.txt"
value2cate = {int(data_pre.cate2value[key])+1:key for key in data_pre.cate2value}
value2cate[0] = ["pad_token"]

gen_data_iter = GenDataIter(data_path=data_pre.get_data_path(), batch_size=1000)
generator = Generator(config)

gen_criterion = nn.NLLLoss(reduction='sum')
gen_optimizer = optim.Adam(generator.parameters())

if config["use_cuda"]:
    generator = generator.cuda()
    gen_criterion = gen_criterion.cuda()
    
def generate_samples(model, batch_size, generated_num, output_file, value2cate):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, config["max_push_len"]).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(value2cate[s]) for s in sample])
            fout.write('%s\n' % string)
            
def eval_diversity(model, batch_size, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, config["max_push_len"]).cpu().data.numpy().tolist()
        samples.extend(sample)
    num_diversity = 0
    for sample in samples:
        if 0 in sample:
            num_diversity += len(set(sample)) - 1
        else:
            num_diversity += len(set(sample))
    return num_diversity / len(samples)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_cates = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = V(data)
        target = V(target)
        if config["use_cuda"]:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        print("loss per batch: ", loss.item())
        total_loss += loss.item()
        total_cates += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_cates)

print('Pretrain with MLE ...')
for epoch in range(PRE_EPOCH_NUM):
    loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
    print('Epoch [%d] Model Loss: %f'% (epoch, loss))
    generate_samples(generator, 1000, 2000, OUT_FILE, value2cate)
    

# train with evaluator!
print('#####################################################')
print('Start EVA2GEN Training...\n')
TOTAL_BATCH = 20
BATCH_SIZE = 1000
rollout = Rollout(generator, 0.8)

evaluator = t.load(PRE_EVA_PATH)

gen_gan_loss = GANLoss()
gen_gan_optm = optim.Adam(generator.parameters())
if config["use_cuda"]:
    gen_gan_loss = gen_gan_loss.cuda()
gen_criterion = nn.NLLLoss(reduction='sum')
if config["use_cuda"]:
    gen_criterion = gen_criterion.cuda()
    
for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, config["max_push_len"])
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = t.zeros((BATCH_SIZE, 1)).type(t.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = V(t.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = V(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, evaluator)
            print("Rewards: ", sum(sum(rewards))/10000)
            rewards = V(t.Tensor(rewards))
            rewards = t.exp(rewards).contiguous().view((-1,))
            if config["use_cuda"]:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()


        rollout.update_params() 
#generate samples for check the diversity:
final_sample_file = "F:/final_samples.txt"
generate_samples(generator, 1000, 2000, final_sample_file, value2cate)
diversity = eval_diversity(generator, 1000, 2000)

t.save(generator, PRE_GEN_PATH)
    
    




###################################################################







