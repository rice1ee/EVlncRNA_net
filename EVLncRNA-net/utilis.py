import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from functools import partial

import torch
import torch_geometric.transforms as T
import torch_geometric.utils as ut
from torch_geometric.data import Data



class BipartiteData(Data):


    def _add_other_feature(self, other_feature):
        self.other_feature = other_feature

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value, *args, **kwargs)
    @property
    def num_nodes(self):
        return self.x_src.size(0) + self.x_dst.size(0)


class GraphDataset():

    def __init__(self, pnode_feature, fnode_feature, other_feature, edge, graph_label):
        self.pnode_feature = pnode_feature
        self.fnode_feature = fnode_feature
        self.other_feature = other_feature
        self.edge = edge
        self.graph_label = graph_label

    def process(self):
        data_list = []  # graph classification need to define data_list for multiple graph
        for i in range(self.pnode_feature.shape[0]):
            edge_index = torch.tensor(self.edge, dtype=torch.long)  # edge_index should be long type

            x_p = torch.tensor(self.pnode_feature[i, :, :], dtype=torch.float)
            x_f = torch.tensor(self.fnode_feature[i, :, :], dtype=torch.float)
            if type(self.graph_label) == np.ndarray:
                y = torch.tensor([self.graph_label[i]], dtype=torch.long)
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y,num_nodes=None)
            else:
                data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, num_nodes=None)

            if type(self.other_feature) == np.ndarray:
                other_feature = torch.tensor(self.other_feature[i, :,:], dtype=torch.float)
                data._add_other_feature(other_feature)

            data_list.append(data)

        return data_list


class Biodata:
    def __init__(self, fasta_file, label_file=None, feature_file=None, K=3, d=3):
        self.dna_seq = {}
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            self.dna_seq[seq_record.id] = str(seq_record.seq)

        if feature_file == None:
            self.other_feature = None
        else:
            self.other_feature = np.load(feature_file)['one']

        self.K = K
        self.d = d

        self.edge = []
        for i in range(4 ** (K * 2)):
            a = i // 4 ** K
            b = i % 4 ** K
            self.edge.append([a, i])
            self.edge.append([b, i])
        self.edge = np.array(self.edge).T

        if label_file:
            self.label = np.loadtxt(label_file)
        else:
            self.label = None

    def encode(self, thread=10):
        print("Encoding sequences...")
        seq_list = list(self.dna_seq.values())
        pool = Pool(thread)
        partial_encode_seq = partial(matrix_encoding, K=self.K, d=self.d)
        feature = np.array(pool.map(partial_encode_seq, seq_list))
        pool.close()
        pool.join()
        self.pnode_feature = feature.reshape(-1, self.d, 4 ** (self.K * 2))
        self.pnode_feature = np.moveaxis(self.pnode_feature, 1, 2)
        zero_layer = feature.reshape(-1, self.d, 4 ** self.K, 4 ** self.K)[:, 0, :, :]
        self.fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4 ** self.K, 1)
        del zero_layer

        graph = GraphDataset(self.pnode_feature, self.fnode_feature, self.other_feature, self.edge, self.label)
        dataset = graph.process()

        return dataset



import numpy as np
import numpy as np




def _num_transfer(seq):
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    seq = ''.join(list(filter(str.isdigit, seq)))

    return seq


def _num_transfer_loc(num_seq,K):
    loc = []
    for i in range(0, len(num_seq)-K+1):
        loc.append(int(num_seq[i:i+K], 4))
    return loc

def _loc_transfer_matrix(loc_list, dis_list,K,length):
    matrix = np.zeros((4**K, 4**K))
    num = 0
    for dis in dis_list:
        for i in range(0, len(loc_list)-K-dis):
            matrix[loc_list[i]][loc_list[i+K+dis]] += 1
        num = num + (length - 2*K - dis + 1.0)

    matrix = matrix / num

    new_matrix = matrix.flatten()

    return new_matrix

def _matrix_encoding(seq,K,d):
    seq = seq.upper()
    length = len(seq)
    num_seq = _num_transfer(seq)
    loc = _num_transfer_loc(num_seq, K)
    dis = [list(range(0, 1)), list(range(1, 2)), list(range(2, 3)),
            list(range(3, 5)), list(range(5, 9)), list(range(9, 17)), list(range(17, 33)),
            list(range(33, 65))]
    if d == 1:
        feature = np.hstack((_loc_transfer_matrix(loc, list(range(0, 1)), K, length)))

    elif d == 2:
        feature = np.hstack((
            _loc_transfer_matrix(loc, list(range(0, 1)), K, length),
            _loc_transfer_matrix(loc, list(range(1, 2)), K, length)))
    else:
        feature = np.hstack((
            _loc_transfer_matrix(loc, list(range(0, 1)), K, length),
            _loc_transfer_matrix(loc, list(range(1, 2)), K, length)))
        for i in range(2, d):
            feature = np.hstack((feature, _loc_transfer_matrix(loc, dis[i], K, length)))

    return feature * 100

def matrix_encoding(seq, K, d):

    return _matrix_encoding(seq, K, d)



