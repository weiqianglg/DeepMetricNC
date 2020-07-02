import logging
import random

import networkx as nx
import numpy as np
import torch

from Dataset import Data


from torch_geometric.utils import to_undirected


class SplitGraph(object):
    def __init__(self, data, train_edge_ratio=0.8):
        self.data = data
        self.num_nodes = self.data.x.size(0)

        logging.info("graph node feature {:d}".format(self.data.x.size(1)))

        self.graph = self.make_graph()

        self.train_edge_ratio = train_edge_ratio

        self.observed_graph = self.get_observed_graph()
        self.observed_index = self.observed_graph.number_of_nodes()

        self.reorder_node()
        self.store_graph()

        self.train_mask = None
        self.train_pos_edge_index = self.test_pos_edge_index = None
        self._train_neg_edge_index = self._test_neg_edge_index = None

        self.train_pos_edge_mask = None

        self.split()

    def __relabel_graph(self, x, node, index_start=0):
        """hepler func for reorder_node"""
        for i, n in enumerate(node, start=index_start):
            x[i] = self.data.x[n]
            self.graph.node[n]["index"] = i
            if i < self.observed_index:
                self.observed_graph.node[n]["index"] = i

    def __rebuild_graph(self, g):
        """hepler func for reorder_node"""
        _graph = nx.empty_graph()
        _graph.add_nodes_from([g.node[u]["index"] for u in g.nodes])
        _graph.add_edges_from([
            (g.node[u]["index"], g.node[v]["index"]) for u, v in g.edges
        ])
        return _graph

    def reorder_node(self):
        x = torch.empty_like(self.data.x)

        observed_node = list(self.observed_graph.nodes)
        self.__relabel_graph(x, observed_node, 0)
        left_node = set(list(self.graph.nodes)) - set(observed_node)
        self.__relabel_graph(x, left_node, self.observed_index)

        self.observed_graph = self.__rebuild_graph(self.observed_graph)
        self.graph = self.__rebuild_graph(self.graph)

        all_edge = [e for e in self.graph.edges]
        self.data = Data(x, torch.tensor(all_edge).transpose(0, 1))

        logging.info("reorder graph, make observed graph at left-up corner of the adj matrix")

    def store_graph(self, observed_graph_name="observed.edgelist", complete_graph_name="all.edgelist"):
        nx.write_edgelist(nx.to_directed(self.observed_graph), observed_graph_name, data=False, delimiter='\t')
        nx.write_edgelist(nx.to_directed(self.graph), complete_graph_name, data=False, delimiter='\t')
        logging.info(f"write observed graph to {observed_graph_name} and complete graph to {complete_graph_name}")


    def split(self):
        # prepare pos/neg train/test edges
        self.train_mask = self.set_train_mask()
        self.set_pos_edge()
        self.train_pos_edge_mask = self.set_train_pos_edge_mask()
        self.set_neg_edge()

    def make_graph(self):
        g = nx.Graph()
        g.add_nodes_from(list(range(self.num_nodes)))

        row, col = self.data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        edge_list = [(row[i].item(), col[i].item()) for i in range(row.shape[0])]
        g.add_edges_from(edge_list)

        logging.info("data->graph done. %d nodes with %d edges." % (g.number_of_nodes(), g.number_of_edges()))
        return g

    def get_observed_graph(self):
        """randomly pick a connected subgraph using BFS."""
        train_edge_number = int(self.graph.number_of_edges() * self.train_edge_ratio)
        added_node = set()
        added_edges_number = 0

        _node = list(self.graph.nodes)
        start_node = random.choice(_node)
        added_node.add(start_node)
        logging.debug("random choose start node {}".format(start_node))

        for p, child in nx.bfs_successors(self.graph, start_node):
            for n in child:
                neighbor_n = set(self.graph.neighbors(n))
                added_edges_number += len(neighbor_n & added_node)
                added_node.add(n)
                if added_edges_number >= train_edge_number:
                    h = self.graph.subgraph(added_node)
                    logging.info("random sample subgraph done. %d edges sampled. with %d nodes" % (h.number_of_edges(),h.number_of_nodes()))
                    return h

        raise RuntimeError("can not get {:d} edges starting from node {:d}".format(train_edge_number, start_node))

    def set_train_mask(self):
        row = col = list(range(self.observed_index))
        train_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        for r in row:
            train_mask[r, col] = True
        return train_mask

    def set_train_pos_edge_mask(self):
        row, col = self.train_pos_edge_index
        pos_mask = torch.zeros(self.observed_index, self.observed_index, dtype=torch.bool)
        pos_mask[row, col] = True
        pos_mask[col, row] = True
        return pos_mask

    def get_train_pos_edge_index(self, duplicated=False):
        if not duplicated:
            return self.train_pos_edge_index
        else:
            return to_undirected(self.train_pos_edge_index)

    def set_pos_edge(self):
        train_pos_edge_index = [e for e in self.observed_graph.edges]
        self.train_pos_edge_index = torch.tensor(train_pos_edge_index).transpose(0, 1)
        logging.info("set train pos edge done.")

        test_pos_edge_index = [e for e in self.graph.edges if e not in self.observed_graph.edges]
        self.test_pos_edge_index = torch.tensor(test_pos_edge_index).transpose(0, 1)
        logging.info("set test pos edge done.")

    def set_neg_edge(self):
        """only set all neg row and col, but leave actual edges later"""
        neg_row, neg_col = self.generate_neg_edge_index(self.train_pos_edge_index, ~self.train_mask)
        self._train_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
        logging.info("set train neg edge done.")

        neg_row, neg_col = self.generate_neg_edge_index(self.test_pos_edge_index, self.train_mask)
        self._test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)
        logging.info("set test neg edge done.")

    def generate_neg_edge_index(self, pos_edge_index, fake_pos_mask):
        """generate all neg edges, except pos_edge_index==1, fake_pos_mask==1"""
        row, col = pos_edge_index
        pos_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.uint8)
        pos_mask[row, col] = 1
        pos_mask = pos_mask.triu(diagonal=1)
        pos_mask = pos_mask.to(torch.bool)

        fake_pos_mask = fake_pos_mask.to(torch.uint8)
        fake_pos_mask = fake_pos_mask.triu(diagonal=1)
        fake_pos_mask = fake_pos_mask.to(torch.bool)

        neg_adj_mask = torch.ones(self.num_nodes, self.num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[fake_pos_mask] = 0
        neg_adj_mask[pos_mask] = 0
        neg_row, neg_col = neg_adj_mask.nonzero().t()
        return torch.stack([neg_row, neg_col], dim=0)

    def get_neg_edge_helper(self, neg_edge_index, edge_num):
        neg_edge_num = neg_edge_index.size(1)
        edge_num = min(edge_num, neg_edge_num)
        perm = torch.tensor(random.sample(range(neg_edge_num), edge_num))
        perm = perm.to(torch.long)
        r = neg_edge_index[:, perm]
        neg_edge_index = r
        return neg_edge_index

    def get_train_neg_edge(self):
        return self.get_neg_edge_helper(self._train_neg_edge_index,
                                        self.observed_graph.number_of_edges())

    def get_test_neg_edge(self):
        return self.get_neg_edge_helper(self._test_neg_edge_index,
                                        self.graph.number_of_edges() - self.observed_graph.number_of_edges())

    def get_train_neg_edge_by_semi_mining(self, distance_z):
        import time
        ts = time.perf_counter()
        distance_z = distance_z[:self.observed_index, :self.observed_index]
        result = []
        sorted_dis, index_dis = torch.sort(distance_z, descending=True)
        for i, row in enumerate(index_dis):
            positive_count = 0
            pos_mask = self.train_pos_edge_mask[i]
            for j in row:
                j = j.item()
                if pos_mask[j]:
                    positive_count += 1
                elif positive_count > 0:
                    result.extend([(i, j)] * positive_count)
                    positive_count = 0
        result = torch.tensor(result).t()
        te = time.perf_counter()
        print("get_train_neg_edge_by_semi_mining", te - ts)
        return result.to(distance_z.device)

    def get_train_neg_edge_by_hard_mining(self, distance_z):
        import time
        ts = time.perf_counter()
        distance_z = distance_z[:self.observed_index, :self.observed_index]
        result = []
        sorted_dis, index_dis = torch.sort(distance_z, descending=True)
        for i, row in enumerate(index_dis):
            positive_count = 0
            pos_mask = self.train_pos_edge_mask[i]
            for j in row:
                j = j.item()
                if pos_mask[j]:
                    positive_count += 1
                elif positive_count > 0:
                    result.extend([(i, j)] * positive_count)
                    positive_count = 0
        result = torch.tensor(result).t()
        te = time.perf_counter()
        print("get_train_neg_edge_by_semi_mining", te - ts)
        return result.to(distance_z.device)


    def to(self, dev):
        self.data = Data(self.data.x.to(dev), self.data.edge_index.to(dev))
        self.train_mask = self.train_mask.to(dev)
        self.train_pos_edge_mask = self.train_pos_edge_mask.to(dev)
        self.train_pos_edge_index = self.train_pos_edge_index.to(dev)
        self.test_pos_edge_index = self.test_pos_edge_index.to(dev)
        self._train_neg_edge_index = self._train_neg_edge_index.to(dev)
        self._test_neg_edge_index = self._test_neg_edge_index.to(dev)

    def save_npz(self, filename, **kwargs):
        datax = self.data.x.numpy()
        edge_index = self.data.edge_index.numpy()
        observed_index = np.array(self.observed_index)
        kwargs.update({'datax': datax, 'edge_index': edge_index,
                       'observed_index': observed_index})
        np.savez(filename, **kwargs)

    def get_adj_matrix(self, edge_index):
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        row, col = edge_index
        adj[row, col] = 1
        return adj


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%H-%M-%S", format="%(asctime)s %(message)s")
    from Dataset import Cora, Test

    random.seed(12345)
    data, _ = Test()
    dg = SplitGraph(data, 0.75)

    # distance_z = torch.rand(dg.num_nodes, dg.num_nodes)
    # for i in range(10):
    #     dg.get_train_neg_edge_by_semi_mining(distance_z)

