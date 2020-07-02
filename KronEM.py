import snap
import networkx as nx


class KronEM:
    def __init__(self, base_path, n_samples):
        self.base_path = base_path
        self.n_samples = n_samples
        self.g_list = []

        self.prepare_graphs()

    def get_possibility(self, u, v):
        cnt = 0
        for g in self.g_list:
            if g.has_edge(u, v):
                cnt += 1
        return cnt / self.n_samples

    def prepare_graphs(self):
        for n in range(self.n_samples):
            LEdgeV, NodePerm = self.load_z_deta(n)
            g = self.get_true_graph(LEdgeV, NodePerm)
            self.g_list.append(g)

    def load_z_deta(self, index=0):
        LEdgeVIn = snap.TFIn(f"{base_path}\\LEdgeV-{index}.bin")
        LEdgeV = snap.TIntTrV()
        LEdgeV.Load(LEdgeVIn)

        NodePermIn = snap.TFIn(f"{base_path}\\NodePerm-{index}.bin")
        NodePerm = snap.TIntV()
        NodePerm.Load(NodePermIn)

        return LEdgeV, NodePerm

    @staticmethod
    def build_node_map(NodePerm):
        node_map = {}
        for i in range(NodePerm.Len()):
            node_map[i] = NodePerm[i]
        return node_map

    def get_true_graph(self, LEdgeV, NodePerm):
        g = nx.Graph()
        node_map = self.build_node_map(NodePerm)
        for i in range(LEdgeV.Len()):
            x, y = LEdgeV[i].GetVal1(), LEdgeV[i].GetVal2()
            x, y = node_map[x], node_map[y]
            g.add_edge(x, y)
        return g


if __name__ == '__main__':
    base_path = r"S:\wq\Snap-5.0\examples\kronem"
    kron = KronEM(base_path, 20)
    p = kron.get_possibility(1,2)
    print(p)
