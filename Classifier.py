import logging
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score, average_precision_score

class SvmClassifier(object):
    TRAIN_ITEM = 1
    def __init__(self, data_graph):
        self.data_graph = data_graph

    def __generate_X_y(self, edge_index, label=True, exchange=True):
        """helper func, generate X->y for training and testing."""
        row, col = edge_index  # edge_index must be undirected
        if exchange:
            row, col = col, row
        x_row, x_col = self.data_graph.data.x[row], self.data_graph.data.x[col]
        x_row, x_col = x_row.numpy(), x_col.numpy()
        X = np.hstack((x_row, x_col))
        y = np.ones(X.shape[0]) if label else np.zeros(X.shape[0])
        return X, y

    def get_train_data(self):
        all_pos_X = np.empty((0, 2*self.data_graph.data.x.size(1)))
        all_neg_X = np.empty((0, 2*self.data_graph.data.x.size(1)))
        all_pos_y = np.empty(0)
        all_neg_y = np.empty(0)
        exchange = False
        for _ in range(self.TRAIN_ITEM):
            pos_X, pos_y = self.__generate_X_y(self.data_graph.train_pos_edge_index, True, exchange)
            all_pos_X = np.vstack((all_pos_X, pos_X))
            all_pos_y = np.hstack((all_pos_y, pos_y))
            neg_X, neg_y = self.__generate_X_y(self.data_graph.get_train_neg_edge(), False, exchange)
            all_neg_X = np.vstack((all_neg_X, neg_X))
            all_neg_y = np.hstack((all_neg_y, neg_y))
            exchange = ~exchange
        return np.vstack((all_pos_X, all_neg_X)), np.hstack((all_pos_y, all_neg_y))

    def get_test_data(self):
        pos_X, pos_y = self.__generate_X_y(self.data_graph.test_pos_edge_index, True)
        neg_X, neg_y = self.__generate_X_y(self.data_graph.get_test_neg_edge(), False)
        return np.vstack((pos_X, neg_X)), np.hstack((pos_y, neg_y))

    def run(self):
        X, y = self.get_train_data()
        classifier = svm.LinearSVC()
        classifier.fit(X, y)
        test_X, test_y = self.get_test_data()
        pred = classifier.predict(test_X)
        auc, ap = roc_auc_score(test_y, pred), average_precision_score(test_y, pred)
        length = len(pred)//2
        mae_pos = np.sum(np.ones(length) - pred[:length]) / length
        mae_neg = np.sum(pred[length:] - np.zeros(length)) / length
        logging.info('SVM AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
        return auc, ap, mae_pos, mae_neg

def run(seed=123, ratio=0.1, dataset_s='Cora'):
    import random
    random.seed(seed)
    from Dataset import Test, Pubmed, Cora, Citeseer, concat_label
    dataset = {"Pubmed": Pubmed, "Cora": Cora, "Citeseer": Citeseer}
    from DataGraph import SplitGraph
    data, y = dataset[dataset_s]()
    data = concat_label(data, y)
    data_graph = SplitGraph(data, train_edge_ratio=ratio)
    f = SvmClassifier(data_graph)
    auc, ap, mae_pos, mae_neg = f.run()
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
        if len(auc_results) >= 10:
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
        tocsv(f"./result/SVC{dataset}AUC.csv", auc_results, ratio)
        tocsv(f"./result/SVC{dataset}AP.csv", ap_results, ratio)
        tocsv(f"./result/SVC{dataset}MAEP.csv", mae_pos_results, ratio)
        tocsv(f"./result/SVC{dataset}MAEN.csv", mae_neg_results, ratio)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%M-%S", format="%(asctime)s %(message)s")
    # main('Cora')
    # main('Citeseer')
    main('Pubmed')
