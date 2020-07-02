import logging
import random
import networkx as nx
import numpy as np
import scipy.linalg
import sklearn.metrics.pairwise as pdist
from sklearn.metrics import roc_auc_score, average_precision_score


class Forsati(object):
    def __init__(self, data_graph):
        self.data_graph = data_graph
        self.observed_index = data_graph.observed_index
        self.graph_matrix = self.get_graph_matrix()
        self.similarity_matrix = self.get_feature_similarity_matrix(self.data_graph.data.x)
        # self.similarity_matrix = self.graph_matrix + np.eye(self.data_graph.num_nodes)

    def get_graph_matrix(self):
        graph_matrix = nx.adjacency_matrix(self.data_graph.graph).todense()
        logging.info("graph matrix done.")
        return np.array(graph_matrix)

    @staticmethod
    def get_feature_similarity_matrix(data_x):
        feature_similarity = pdist.pairwise_distances(data_x, metric="cosine", n_jobs=8)
        feature_similarity = 1 - feature_similarity  # / np.max(feature_similarity)
        logging.info("feature similarity matrix done.")
        return feature_similarity

    def get_Us(self, top_s):
        e_vals, e_vecs = scipy.linalg.eigh(self.similarity_matrix)
        sorted_indices = np.argsort(e_vals)
        logging.info("eig done.")
        Us = e_vecs[:, sorted_indices[:-top_s - 1:-1]]
        return Us

    def get_U_s(self, Us):
        row = random.sample(range(self.data_graph.num_nodes), self.observed_index)
        row = list(range(self.observed_index))
        return Us[row, :]

    def get_O(self):
        return self.graph_matrix[0:self.observed_index, 0:self.observed_index]

    def get_A_(self, top_s=None):
        # np.save("feature_similarity_matrix.npy", self.feature_similarity_matrix)
        if top_s is None:
            top_s = 20  # self.graph_matrix.ndim + 8
            logging.info(f"set top_s as rank of graph matrix, {top_s}")
        Us = self.get_Us(top_s)
        U_s = self.get_U_s(Us)

        # pinv = np.linalg.pinv(U_s.T @ U_s, rcond=1e-40)
        pinv = scipy.linalg.pinvh(U_s.T @ U_s, check_finite=False)

        # y = np.allclose(pinv, pinv.T, atol=0.01)
        # print("is close.", y)
        A_ = pinv @ U_s.T @ self.get_O() @ U_s @ pinv

        return Us @ A_ @ Us.T

    def get_scores(self, A_):
        A_c = A_.copy().real
        A_c[A_c < 0] = 0
        A_c[A_c > 1] = 1

        row, col = self.data_graph.test_pos_edge_index
        pos_edge_pred = A_c[row, col]

        row, col = self.data_graph.get_test_neg_edge()
        neg_edge_pred = A_c[row, col]

        testY = np.hstack((np.ones(len(pos_edge_pred)), np.zeros(len(neg_edge_pred))))
        pred = np.hstack((pos_edge_pred, neg_edge_pred))

        mae_pos = np.sum(np.ones(len(pos_edge_pred)) - pos_edge_pred) / len(pos_edge_pred)
        mae_neg = np.sum(neg_edge_pred - np.zeros(len(neg_edge_pred))) / len(neg_edge_pred)

        return roc_auc_score(testY, pred), average_precision_score(testY, pred), mae_pos, mae_neg


def run(seed=123, ratio=0.1, dataset_s='Cora'):
    import random
    random.seed(seed)
    from Dataset import Test, Pubmed, Cora, Citeseer, concat_label
    dataset = {"Pubmed": Pubmed, "Cora": Cora, "Citeseer": Citeseer}
    from DataGraph import SplitGraph
    data, y = dataset[dataset_s]()
    data = concat_label(data, y)
    data_graph = SplitGraph(data, train_edge_ratio=ratio)
    f = Forsati(data_graph)
    A_ = f.get_A_()
    logging.info("A_ max %f, min %f, mean %f." % (A_.max(), A_.min(), A_.mean()))
    auc, ap, mae_pos, mae_neg = f.get_scores(A_)
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
        if len(auc_results) >= 50:
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
        tocsv(f"./result/Forsati{dataset}AUC.csv", auc_results, ratio)
        tocsv(f"./result/Forsati{dataset}AP.csv", ap_results, ratio)
        tocsv(f"./result/Forsati{dataset}MAEP.csv", mae_pos_results, ratio)
        tocsv(f"./result/Forsati{dataset}MAEN.csv", mae_neg_results, ratio)


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    main('Cora')
