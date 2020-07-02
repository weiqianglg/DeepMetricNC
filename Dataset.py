import os.path as osp
from collections import namedtuple
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import networkx as nx


"""all the dataset will return feature tensor and edge_index tensor."""
Data = namedtuple('Data', ['x', 'edge_index'])


def concat_label(data, y):
    x = data.x
    unique_y = torch.unique(y)
    unique_y_length = unique_y.size(0)
    code_y = torch.eye(unique_y_length)
    x_with_label = torch.cat((code_y[y], x), dim=1)
    return Data(x=x_with_label, edge_index=data.edge_index)


def Test():
    g = nx.read_edgelist(r"D:\project\snap-master\examples\Release\graph.txt", delimiter='\t', nodetype=int, data=False)
    g = nx.fast_gnp_random_graph(10, 0.5)
    x = torch.eye(g.number_of_nodes())
    edge = [e for e in g.edges]
    edge_index = to_undirected(torch.tensor(edge).transpose(0, 1))
    return Data(x, edge_index), None

def Cora():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Cora')
    dataset = Planetoid(path, 'Cora', T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y


def Pubmed():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Pubmed')
    dataset = Planetoid(path, 'Pubmed', T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y


def Citeseer():
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Citeseer')
    dataset = Planetoid(path, 'Citeseer', T.NormalizeFeatures())
    data = dataset[0]
    data_ = Data(x=data.x, edge_index=data.edge_index)
    return data_, data.y


if __name__ == '__main__':
    Citeseer()



