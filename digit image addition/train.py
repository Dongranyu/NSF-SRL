
import argparse
import json
import os
import sys
import pickle
import resource
from tkinter import N
import traceback
import logging
from collections import defaultdict
from typing import Counter
import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
import pdb
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#import _init_paths  # pylint: disable=unused-import
from os.path import join as joinpath
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import new_dataset as Dataset
from new_mln import Posterior
from add_network import Separate_Baseline
#from network import Network
# Set up logging and load config options
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def top_down(net_pre, rel_prob):

    # score = torch.dist(sub_score, x_prob.view(1,-1), p=2)+ \
    #         torch.dist(obj_score, y_prob.view(1, -1), p=2)+ \
    #         torch.dist(pred_score, rel_prob.view(1, -1), p=2)
    # score = F.pairwise_distance(sub_score,x_prob.view(1,-1),p=2)+\
    #     F.pairwise_distance(obj_score,y_prob.view(1,-1),p=2)+\
    #     F.pairwise_distance(pred_score,rel_prob.view(1,-1),p=2)
    # loss = torch.nn.L1Loss()
    # score = loss(sub_score,x_prob.flatten())+ \
    #         loss(obj_score, y_prob.flatten())+ \
    #         loss(pred_score, rel_prob.flatten())

    #distance
    loss = torch.nn.MSELoss(reduce=True, size_average=True)
    score = loss(net_pre, rel_prob)
    
    #kl
    
    # score=F.kl_div(F.log_softmax(sub_score),F.softmax(x_prob,dim=-1),reduction='sum') + \
    #       F.kl_div(F.log_softmax(obj_score),F.softmax(y_prob,dim=-1),reduction='sum') + \
    #       F.kl_div(F.log_softmax(pred_score),F.softmax(rel_prob,dim=-1),reduction='sum')


    return score


    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
  

def main():
    """Main function"""
    print(os.path.abspath('.'))
    data_root = os.path.abspath(".") + "/data/MNIST/mln_data"
    mln_dataset = Dataset(data_root)
    pred_txt = joinpath(data_root, "new_pred.json")
    posterior = Posterior(256, 5, pred_txt).cuda()
    
    n_epochs = 3            
    batch_size_train = 64   
    batch_size_test = 1000          
    log_interval = 100       
    random_seed = 1         
    torch.manual_seed(random_seed)
    
    train_data=np.load("/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/train_image_features.npy")
    train_label=np.load("/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/train_image_labels.npy")
    ent_pair=[]#record id of entities
    with open(joinpath("/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/","fact.txt")) as f:
            for line in f:
                relation_id = line.strip().split()[1]
                ent1_id, ent2_id = line.strip().split()[0], line.strip().split()[2]
                ent_pair.append((ent1_id,relation_id,ent2_id))
    train_data=train_data.tolist()
    for i in range(len(train_data)):
        train_data[i].append(ent_pair[i])
    train_data=np.array(train_data)
    def batch_generator(all_data, batch_size_train, shuffle=True):
        
        all_data = [np.array(d) for d in all_data]
        #ent_pair =[np.array(k) for k in ent_pair]
        # 获取样本大小
        data_size = all_data[0].shape[0]
        print("data_size: ", data_size)
        if shuffle:
            # 随机生成打乱的索引
            p = np.random.permutation(data_size)
            # 重新组织数据
            all_data = [d[p] for d in all_data]
        batch_count = 0
        while True:
            
            if batch_count * batch_size_train + batch_size_train > data_size:
                batch_count = 0
                if shuffle:
                    p = np.random.permutation(data_size)
                    all_data = [d[p] for d in all_data]
            start = batch_count * batch_size_train
            end = start + batch_size_train
            batch_count += 1
            yield [d[start: end] for d in all_data]
   
    network = Separate_Baseline()
    #network.load_state_dict(torch.load("models/pretrained/all_{}.pth")
    network=network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    
    # all_params = chain.from_iterable([network.parameters(), posterior.parameters()])
    # optimizer = optim.Adam(all_params, lr=0.01, weight_decay=0.0001)
    
    train_losses = []
    count=train_data.shape[0]
    batch_num = count // batch_size_train
    batch_gen= batch_generator([train_data, train_label], batch_size_train)
    
    def train(epoch):
        network.train()
        
        for batch_idx in range(batch_num):
            batch_x, batch_y = next(batch_gen)
            #print("batch_x",batch_x[0])
            print('batch_y',batch_y)
            #pdb.set_trace()
            batch_x1=[]
            batch_x2=[]
            triples=[]
            for k in batch_x:
                batch_x1.append(k[0])#initial fearure of ent1
                batch_x2.append(k[1])#initial fearure of ent2
                triples.append(k[2])#(ent1,relation,ent2)
           
            batch_x1=torch.Tensor(batch_x1).view(batch_size_train,28,28)
            batch_x2=torch.Tensor(batch_x2).view(batch_size_train,28,28)
            batch_y=torch.from_numpy(batch_y)#[64]
            output,embed = network(batch_x1.to(device),batch_x2.to(device))
           
            net_loss = F.nll_loss(output, batch_y.to(device))   
            predictions = torch.argmax(output, dim = 1)
            #predictions = output.data.max(1, keepdim=True)[1]
            #print(predictions)
            train_accuracy = 100*torch.sum(predictions == batch_y.to(device))/batch_y.shape[0]
            triple_ent1=[]
            triple_ent2=[]
            for n in range(len(triples)):
                triple_ent1.append(triples[n][0]) 
                triple_ent2.append(triples[n][2])   
            emb_dic1=dict(zip(triple_ent1,embed[0]))
            emb_dic2=dict(zip(triple_ent2,embed[1]))
            emb_dic = dict( emb_dic1, **emb_dic2 )
            pdb.set_trace()
            sub=triples
            mln_acc_loss = 0.0
            samples_by_r, neg_mask_by_r, obs_by_r, neg_by_r, rids = mln_dataset.find_rule(sub)
            mln_loss = 0.0
            cnt = 0
            pre_M = []
            latent = []
           
            #E-step begin
            print('========================E-step=============================')
            for ind, samples in enumerate(samples_by_r):
                neg_mask = neg_mask_by_r[ind]
                obs_var = obs_by_r[ind]
                neg_var = neg_by_r[ind]
                if sum(len(v[1]) for v in neg_mask) == 0:
                    continue
                lat = neg_var[1][1][0]
                latent.append(lat)
                
                rule_score,atom_loss,hinge_loss,x_prob,y_prob,pred_prob,latent_score,pred_score,latent_ = posterior(emb_dic,[samples,neg_mask,obs_var,neg_var])
                pre_M.append(torch.sigmoid(pred_score))
                
                #sub_ = samples[2][1]
                link_loss = top_down(output[ind], pred_prob)
                
                mln_loss += - (rule_score.sum()  + latent_score)+ link_loss + atom_loss + hinge_loss
                
                cnt += 1
            if cnt > 0:
                mln_loss /= cnt
                mln_acc_loss += mln_loss.item()
            loss = net_loss +  mln_acc_loss
         
            
        
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},Accuracy:{}\n'.format(epoch, batch_idx * len(batch_x),
                                                                            len(train_data),
                                                                            100.* batch_idx / len(batch_x),
                                                                            loss.item(), train_accuracy))
                train_losses.append(loss.item())
                # train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(network.state_dict(), '/home/yudongran/ydr2/general_framework/model/train_model.pth')
                # torch.save(optimizer.state_dict(), './optimizer.pth')
                
    test_data=np.load("/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/test_image_features.npy")
    test_label=np.load("/home/yudongran/ydr2/general_framework/data/MNIST/mln_data/test_image_labels.npy")
    test_count=test_data.shape[0]
    test_batch_num = test_count // batch_size_test
    test_batch_gen= batch_generator([test_data, test_label], batch_size_test)
    test_losses = []
    #network.load_state_dict(torch.load('/home/yudongran/ydr2/general_framework/model/train_model.pth'))#load saved model
    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():  
            
            for t in range(test_batch_num):
                #pdb.set_trace()
                test_batch_x, test_batch_y = next(test_batch_gen)
                test_batch_x1=torch.from_numpy(test_batch_x[:,0]).view(batch_size_test,28,28)
                test_batch_x2=torch.from_numpy(test_batch_x[:,1]).view(batch_size_test,28,28)
                test_batch_y=torch.from_numpy(test_batch_y)#[64]
                test_output,test_embed = network(test_batch_x1.to(device),test_batch_x2.to(device))
                test_loss += F.nll_loss(test_output, test_batch_y.to(device), reduction='sum').item()
                
                pred = test_output.data.max(1, keepdim=True)[1]
                
                correct += pred.eq(test_batch_y.to(device).data.view_as(pred)).sum()
               
        test_loss /= test_count
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data),
            100. * correct / len(test_data)))


    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
        print("*"*60)
if __name__ == '__main__':
    main()
