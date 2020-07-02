import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score


# This is a basic multilayer perceptron
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=True):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size, bias=False))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        z = self.net(x)
        return z


class SimilarityMetric(object):
    def __init__(self, data_graph, in_channels, out_channels=32, device='cpu'):
        super(SimilarityMetric, self).__init__()
        self.in_channels = in_channels
        device = torch.device(device)
        self.mpl = MLP([in_channels, out_channels]).to(device)
        self.data_graph = data_graph
        self.data_graph.to(device)
        self.optimizer = None
        self.loss_val = 0.0

    @staticmethod
    def inner_product(z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    @staticmethod
    def inner_product_all(z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)

    def loss_f(self, z, pos_edge_index, neg_edge_index):
        eps = 1e-15
        pos_loss = -torch.log(
            self.inner_product(z, pos_edge_index) + eps).mean()
        neg_loss = -torch.log(
            1 - self.inner_product(z, neg_edge_index) + eps).mean()
        return pos_loss + neg_loss

    def train(self):
        self.mpl.train()
        self.optimizer.zero_grad()
        z = self.mpl(self.data_graph.data.x)
        train_neg_edge = self.data_graph.get_train_neg_edge()
        # train_neg_edge = self.data_graph.get_train_neg_edge_by_semi_mining(self.inner_product_all(z))
        loss = self.loss_f(z,
                           self.data_graph.train_pos_edge_index,
                           # self.data_graph.get_train_pos_edge_index(True),
                           train_neg_edge)
        loss.backward(retain_graph=False)
        self.loss_val = loss
        self.optimizer.step()

    def test(self, z, pos_edge_index, neg_edge_index):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.inner_product(z, pos_edge_index)
        neg_pred = self.inner_product(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        pos_pred = pos_pred.detach().cpu().numpy()
        neg_pred = neg_pred.detach().cpu().numpy()
        mae_pos = np.sum(np.ones(len(pos_pred)) - pos_pred) / len(pos_pred)
        mae_neg = np.sum(neg_pred - np.zeros(len(neg_pred))) / len(neg_pred)

        return roc_auc_score(y, pred), average_precision_score(y, pred), mae_pos, mae_neg

    def eval(self, pos_edge_index, neg_edge_index):
        assert pos_edge_index.size(1) == neg_edge_index.size(1)
        self.mpl.eval()
        with torch.no_grad():
            z = self.mpl(self.data_graph.data.x)
        return self.test(z, pos_edge_index, neg_edge_index)

    def run(self):
        self.optimizer = torch.optim.Adam(self.mpl.parameters(), lr=0.01, weight_decay=5e-6)

        max_auc, max_ap, best_epoch = 0, 0, 0
        test_best_epoch = 0
        best_loss = 100

        for epoch in range(1, 1001):
            self.train()
            if self.loss_val < best_loss:
                best_epoch, best_loss = epoch, self.loss_val
                auc, ap, mae_pos, mae_neg = self.eval(self.data_graph.test_pos_edge_index,
                                                      self.data_graph.get_test_neg_edge())
                if auc > max_auc or ap > max_ap:
                    max_auc, max_ap = auc, ap
                    test_best_epoch = epoch
                logging.info(
                    'Epoch: {:03d}, Loss: {:.4f}, auc {:.4f}, ap {:.4f}'.format(epoch, self.loss_val, auc, ap))
            if epoch - best_epoch > 100:
                logging.info("cannot decrease loss within fixed steps, quit.")
                break

        return auc, ap, mae_pos, mae_neg


def run(seed=123, ratio=0.1, dataset_s='Cora'):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    from Dataset import Test, Pubmed, Cora, Citeseer, concat_label
    dataset = {"Pubmed": Pubmed, "Cora": Cora, "Citeseer": Citeseer}
    from DataGraph import SplitGraph

    data, y = dataset[dataset_s]()
    data = concat_label(data, y)
    data_graph = SplitGraph(data, train_edge_ratio=ratio)
    fg = SimilarityMetric(data_graph, data.x.size(1))
    auc, ap, mae_pos, mae_neg = fg.run()
    return auc, ap, mae_pos, mae_neg


def tocsv(name, result, ratio):
    import csv
    with open(name, 'w') as f:
        csv_write = csv.writer(f)
        data_row = ["index"] + [f"{r:.1}" for r in ratio]
        csv_write.writerow(data_row)
        csv_write.writerows(result)

def main(dataset_s):
    dataset = dataset_s
    ratio = [0.1 * i for i in range(1, 10)]
    auc_results, ap_results = [], []
    mae_pos_results, mae_neg_results = [], []
    index, ok = 1, True
    for seed in range(100):
        if len(auc_results) >= 40:
            break
        logging.critical(f"processing {index} round, seed is {seed}")
        auc_r, ap_r = [], []
        mae_p_r, mae_n_r = [], []
        for r in ratio:
            try:
                auc, ap, mae_pos, mae_neg = run(seed, r, dataset)
            except RuntimeError as e:
                auc, ap, mae_pos, mae_neg = None, None, None, None
                break
            auc_r.append(auc)
            ap_r.append(ap)
            mae_p_r.append(mae_pos)
            mae_n_r.append(mae_neg)

        logging.critical(f"finish {index} round at ratio {r}")
        auc_results.append([index] + auc_r)
        ap_results.append([index] + ap_r)
        mae_pos_results.append([index] + mae_p_r)
        mae_neg_results.append([index] + mae_n_r)
        index += 1

        tocsv(f"./result/DML{dataset}AUC.csv", auc_results, ratio)
        tocsv(f"./result/DML{dataset}AP.csv", ap_results, ratio)
        tocsv(f"./result/DML{dataset}MAEP.csv", mae_pos_results, ratio)
        tocsv(f"./result/DML{dataset}MAEN.csv", mae_neg_results, ratio)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    # main('Citeseer')
    main('Cora')
