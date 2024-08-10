'''数据预处理及特征提取'''

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
import itertools
from tqdm import tqdm

def encode_seq(seq):
    """将DNA序列编码为独热编码"""
    seq = seq.upper()
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((len(seq), 4), dtype=np.int8)
    for i, base in enumerate(seq):
        if base in base_dict:
            encoded[i, base_dict[base]] = 1
        else:
            encoded[i, :] = 0.25
    return encoded

def pad_seq(seq, max_length=5000):
    """将独热编码调整为固定长度"""
    encoded_seq = encode_seq(seq)
    length = encoded_seq.shape[0]
    if length > max_length:
        return encoded_seq[:max_length, :]
    else:
        padding = np.zeros((max_length - length, 4), dtype=np.int8)
        return np.vstack((encoded_seq, padding))


def get_fa(dir,maxlen=4000):
    name=dir.split('/')[1]


    with open(dir+name+'_train.fa','r')as f:
        trainx=f.readlines()
        trainx=[x.strip() for x in trainx if '>' not in x]
        onetrainx=[pad_seq(x,maxlen) for x in trainx]

    np.savez(dir+'train_fea.npz',one=onetrainx)


    with open(dir+name+'_test.fa','r')as f:
        testx=f.readlines()
        testx=[x.strip() for x in testx if '>' not in x]
        onetestx=[pad_seq(x,maxlen) for x in testx]
    np.savez(dir+'test_fea.npz',one=onetestx)



if __name__=='__main__':
    for x,y in zip(['human','plant','mouse'],[2000,4000,4000]):
        dir='dataset/{}/'.format(x)
        get_fa(dir,y)
