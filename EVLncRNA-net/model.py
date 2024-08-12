import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler

import copy
import random
import heapq
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
from utilis import Biodata


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num=2, alpha=None, gamma=1., size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class mynet(nn.Module):
    def __init__(self, label_num,other_feature_dim_in=None ,other_feature_dim=None, K=3, d=3, node_hidden_dim=3, gcn_dim=128, gcn_layer_num=4,
                 cnn_dim=64, cnn_layer_num=3, cnn_kernel_size=8, fc_dim=128, dropout_rate=0.2, pnode_nn=True,
                 fnode_nn=True):
        super(mynet, self).__init__()
        self.label_num = label_num
        self.pnode_dim = d
        self.pnode_num = 4 ** (2 * K)
        self.fnode_num = 4 ** K
        self.node_hidden_dim = node_hidden_dim
        self.gcn_dim = gcn_dim
        self.gcn_layer_num = gcn_layer_num
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout_rate
        self.pnode_nn = pnode_nn
        self.fnode_nn = fnode_nn
        self.other_feature_dim = other_feature_dim
        self.other_feature_dim_in=other_feature_dim_in

        self.pnode_d = nn.Linear(self.pnode_num * self.pnode_dim, self.pnode_num * self.node_hidden_dim)
        self.fnode_d = nn.Linear(self.fnode_num, self.fnode_num * self.node_hidden_dim)

        self.gconvs_1 = nn.ModuleList()
        self.gconvs_2 = nn.ModuleList()
        #老鼠的kernel_size第一位数应该为50
        #植物为200

        self.cnn1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(50,4),stride=(1,1),padding=0)
        self.cnn2=nn.Conv2d(in_channels=32,out_channels=128,kernel_size=(50,1),stride=(1,1),padding=0)
        self.batchnorm=nn.BatchNorm2d(128)
        self.pol=nn.AdaptiveMaxPool2d(1)
        self.fcone=nn.Linear(128,self.other_feature_dim)

        self.drop=nn.Dropout(0.3)#植物老鼠0.3  人类
        self.batchnorm2 = nn.BatchNorm1d(256)


        if self.pnode_nn:
            pnode_dim_temp = self.node_hidden_dim
        else:
            pnode_dim_temp = self.pnode_dim

        if self.fnode_nn:
            fnode_dim_temp = self.node_hidden_dim
        else:
            fnode_dim_temp = 1

        for l in range(self.gcn_layer_num):
            if l == 0:
                self.gconvs_1.append(pyg_nn.SAGEConv((fnode_dim_temp, pnode_dim_temp), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, fnode_dim_temp), self.gcn_dim))
            else:
                self.gconvs_1.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))

        self.lns = nn.ModuleList()
        for l in range(self.gcn_layer_num - 1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.convs = nn.ModuleList()
        for l in range(self.cnn_layer_num):
            if l == 0:
                self.convs.append(
                    nn.Conv1d(in_channels=self.gcn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
            else:
                self.convs.append(
                    nn.Conv1d(in_channels=self.cnn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))

        if self.other_feature_dim:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim,
                                self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim + self.other_feature_dim, self.fc_dim + self.other_feature_dim)
            self.d3 = nn.Linear(self.fc_dim + self.other_feature_dim, self.label_num)
        else:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim,
                                self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim, self.label_num)

    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        edge_index_forward = data.edge_index[:, ::2]
        edge_index_backward = data.edge_index[[1, 0], :][:, 1::2]
        if self.other_feature_dim:
            onex=data.other_feature
            onex = torch.reshape(onex, ((int(onex.shape[0]/self.other_feature_dim_in),self.other_feature_dim_in, 4)))
            onex =onex.unsqueeze(3)
            onex = onex.permute(0, 3, 1, 2)
            #print(onex.shape)
            onex = self.cnn1(onex)
            onex = self.cnn2(onex)
            onex = self.batchnorm(onex)
            onex = F.relu(onex)
            onex = self.pol(onex)
            onex = onex.squeeze()
            other_feature = self.fcone(onex)

            # transfer primary nodes
        if self.pnode_nn:
            x_p = torch.reshape(x_p, (-1, self.pnode_num * self.pnode_dim))
            x_p = self.pnode_d(x_p)
            x_p = torch.reshape(x_p, (-1, self.node_hidden_dim))
        else:
            x_p = torch.reshape(x_p, (-1, self.pnode_dim))

        # transfer feature nodes
        if self.fnode_nn:
            x_f = torch.reshape(x_f, (-1, self.fnode_num))
            x_f = self.fnode_d(x_f)
            x_f = torch.reshape(x_f, (-1, self.node_hidden_dim))
        else:
            x_f = torch.reshape(x_f, (-1, 1))

        for i in range(self.gcn_layer_num):
            x_p = self.gconvs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.relu(x_p)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            x_f = self.gconvs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.relu(x_f)
            x_f = F.dropout(x_f, p=self.dropout, training=self.training)
            if not i == self.gcn_layer_num - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)

        x = torch.reshape(x_p, (-1, self.gcn_dim, self.pnode_num))

        for i in range(self.cnn_layer_num):
            x = self.convs[i](x)
            x = F.relu(x)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.other_feature_dim:
            x = x.flatten(start_dim=1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(torch.cat([x, other_feature], 1))
            x=self.batchnorm2(x)
            x=self.drop(x)
            x = F.relu(x)
            out = self.d3(x)
            #out = F.softmax(out, dim=1)

        else:
            x = x.flatten(start_dim=1)
            x = self.d1(x)
            x = F.relu(x)
            out = self.d2(x)
            # out = F.softmax(x, dim=1)

        return out