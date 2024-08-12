# This is an example to train a two-classes model.
import model
import torch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from model import *
import copy
import random
import heapq
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
import utilis
from utilis import Biodata
from model import FocalLoss
from config import Config
from sklearn import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config=Config()

def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=20, random_seed=99, val_split=0.1,
          weighted_sampling=None,iffcloss=None,
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    random.seed(random_seed)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_list if i not in test_list]
    testset = [dataset[i] for i in data_list if i in test_list]

    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [100 / label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)#老鼠植物为True
        train_loader = DataLoader(trainset, batch_size=batch_size, follow_batch=['x_src', 'x_dst'],
                                  sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                  follow_batch=['x_src', 'x_dst'])
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, follow_batch=['x_src', 'x_dst'])


    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)  # 每2个epoch学习率衰减为原来的一半

    # train
    dev_best_loss = float('inf')
    old_test_acc = 0
    if iffcloss==1:
        closs = FocalLoss()
    else:
        closs=torch.nn.CrossEntropyLoss()
    for epoch in range(epoch_n):
        training_running_loss = 0.0
        train_acc = 0.0
        total_batch=0
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            label = batch.y

            # forward + backprop + loss
            pred = model(batch)
            loss = closs(pred, label)
            optimizer.zero_grad()
            loss.backward()

            # update model params
            optimizer.step()

            training_running_loss += loss.detach().item()
            train_acc += (torch.argmax(pred, 1).flatten() == label).type(torch.float).mean().item()

            total_batch=total_batch+1

            msg1 = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}'
            if total_batch%20==0:
                print(msg1.format(total_batch, loss.item(), train_acc/total_batch))

        # test accuracy
        test_acc, dev_loss = evaluation(test_loader, model, device,iffcloss)
        if test_acc > old_test_acc:
            old_test_acc = test_acc
            torch.save(model.state_dict(), config.save_path)#人类取acc最高保存

        if dev_loss < dev_best_loss:

            dev_best_loss = dev_loss
            #torch.save(model.state_dict(), config.save_path)#植物和老鼠数据集取loss最小




        print("Epoch {}| Loss: {:.4f}| Train accuracy: {:.4f}| Validation accuracy: {:.4f}|Validation loss: {:.5f}".format(epoch,
        training_running_loss / (i + 1),train_acc / (i + 1),test_acc,dev_loss))
        scheduler.step()
        print('epoch: ', epoch, 'lr: ', scheduler.get_last_lr())

    return model


def evaluation(loader, model, device,iffcloss):
    model.eval()
    if iffcloss==1:
        closs = FocalLoss()
    else:
        closs=torch.nn.CrossEntropyLoss()
    correct = 0
    loss_total = 0
    b=0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            label = data.y
            loss = closs(pred, label)
            pred = pred.argmax(dim=1)

        correct += pred.eq(label).sum().item()
        loss_total += loss
        b=b+1

    total = len(loader.dataset)
    acc = correct / total

    return acc, loss_total/b


def test(fasta_file,model=None, feature_file=None, label_file=None, output_file="test_output.txt",
         thread=10, K=3, d=3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data = Biodata(fasta_file=fasta_file, feature_file=feature_file, K=K, d=d)
    testset = data.encode(thread=thread)
    model.load_state_dict(torch.load(config.save_path))
    model=model.to(device)

    with open(label_file,'r')as f:
        labels=f.readlines()
        labels=[int(x.strip()) for x in labels]

    allpre=[]
    loader = DataLoader(testset, batch_size=32, shuffle=False, follow_batch=['x_src', 'x_dst'])
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            pred = pred.cpu().numpy().tolist()
            # print(pred)
            for x in pred:
                allpre.append(x)

    report=metrics.classification_report(labels,allpre)
    print(report)
    with open(config.log_dir+'/.report.txt','w',encoding='utf-8')as f:
        f.write(report)
    # f = open(output_file, "w")
    # for each in pred:
    #     f.write(str(each) + "\n")
    # f.close()



if __name__ == '__main__':
    data = Biodata(fasta_file=config.train_x, label_file=config.train_label,
                   feature_file=config.train_fea)
    dataset = data.encode(thread=20)
    #other_feature_dim_in 人类数据集为  老鼠植物为4000 weighted_sampling人类为 植物和老鼠为True random_seed人类 植物老鼠99 other_feature_dim_in人类1000 其他4000
    model = mynet(label_num=2, K=3, d=3, node_hidden_dim=3,other_feature_dim=128,other_feature_dim_in=2000).to(device)
    train(dataset, model, weighted_sampling=False,batch_size=config.batch_size,random_seed=88,
          learning_rate=config.learning_rate,epoch_n=config.num_epochs,iffcloss=1)#iffcloss老鼠植物为1
    #如果只跑测试，只保留167和171行即可，其他的几行可以注释掉，不要忘记在跑之前在model的120 121行把kernel_size=(200,4)的第一位数改过来
    test(model=model,fasta_file=config.test_x,label_file=config.test_label,
                   feature_file=config.test_fea)

