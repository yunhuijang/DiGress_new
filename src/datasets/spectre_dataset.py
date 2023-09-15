import os
import pickle
import torch
from torch.utils.data import random_split, Dataset
from sklearn.model_selection import train_test_split
import networkx as nx
import torch_geometric.utils
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, vstack

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class SpectreGraphDataset(Dataset):
    def __init__(self, data_file):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(
            filename)
        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes)
        return data


class Comm20Dataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('community_12_21_100.pt')


class SBMDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('sbm.pt')


class PlanarDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('planar.pt')
    
class GDSSGraphDataset(SpectreGraphDataset):
    def __init__(self, data_file, adjs):
        if data_file is None:
            self.adjs = adjs
        else:
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
            filename = os.path.join(base_path, data_file)
            with open(filename, 'rb') as f:
                graphs = pickle.load(f)
            self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
            print(f'Dataset {self.data_name} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes)
        return data

    
class ComSmallDataset(GDSSGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'com_small'
        if data_file is None:
            super().__init__(None, adjs=adjs)
        else:
            super().__init__('GDSS_com.pkl', adjs=None)


class EgoDataset(GDSSGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'ego_small'
        if data_file is None:
            super().__init__(None, adjs=adjs)
        else:
            super().__init__('GDSS_ego.pkl', adjs=None)
        
class EnzDataset(GDSSGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'enz'
        if data_file is None:
            super().__init__(None, adjs=adjs)
        else:
            super().__init__('GDSS_enz.pkl', adjs=None)
        
class GridDataset(GDSSGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'grid'
        if data_file is None:
            super().__init__(None, adjs=adjs)
        else:
            super().__init__('GDSS_grid.pkl', adjs=None)

class ProteinsDataset(SpectreGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'proteins'
        if data_file is None:
            self.adjs = adjs
        else:
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
            filename = os.path.join(base_path, data_file)
            self.adjs = self.load_proteins_data(filename)
            # self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
            print(f'Dataset {self.data_name} loaded from file')

    def load_proteins_data(self, data_dir):
    
        min_num_nodes=100
        max_num_nodes=500
        
        adjs = []
        eigvals = []
        eigvecs = []
        n_nodes = []
        n_max = 0
        max_eigval = 0
        min_eigval = 0

        G = nx.Graph()
        # Load data
        path = os.path.join(data_dir, 'proteins/DD')
        data_adj = np.loadtxt(os.path.join(path, 'DD_A.txt'), delimiter=',').astype(int)
        data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)
        data_graph_types = np.loadtxt(os.path.join(path, 'DD_graph_labels.txt'), delimiter=',').astype(int)

        data_tuple = list(map(tuple, data_adj))

        # Add edges
        G.add_edges_from(data_tuple)
        G.remove_nodes_from(list(nx.isolates(G)))

        # remove self-loop
        G.remove_edges_from(nx.selfloop_edges(G))

        # Split into graphs
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1

        for i in tqdm(range(graph_num)):
            # Find the nodes for each graph
            nodes = node_list[data_graph_indicator == i + 1]
            G_sub = G.subgraph(nodes)
            G_sub.graph['label'] = data_graph_types[i]
            if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
                adj = torch.from_numpy(nx.adjacency_matrix(G_sub).toarray()).float()
                L = nx.normalized_laplacian_matrix(G_sub).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)
                
                eigvals.append(eigval)
                eigvecs.append(eigvec)
                adjs.append(adj)
                n_nodes.append(G_sub.number_of_nodes())
                if G_sub.number_of_nodes() > n_max:
                    n_max = G_sub.number_of_nodes()
                max_eigval = torch.max(eigval)
                if max_eigval > max_eigval:
                    max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < min_eigval:
                    min_eigval = min_eigval

        return adjs
    
class LobsterDataset(SpectreGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'lobster'
        if data_file is None:
            self.adjs = adjs
        else:
            graphs = []
            p1 = 0.7
            p2 = 0.7
            count = 0
            min_node = 10
            max_node = 100
            max_edge = 0
            mean_node = 80
            num_graphs = 100

            seed_tmp = 1234
            while count < num_graphs:
                G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
                if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
                    graphs.append(G)
                    if G.number_of_edges() > max_edge:
                        max_edge = G.number_of_edges()
                    count += 1
                seed_tmp += 1
            self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
            print(f'Dataset {self.data_name} loaded from file')

class PointDataset(SpectreGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'point'
        if data_file is None:
            self.adjs = adjs
        else:
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
            filename = os.path.join(base_path, data_file)
            graphs = self.load_point_data(filename, min_num_nodes=0, max_num_nodes=10000, 
                                  node_attributes=False, graph_labels=True)
            self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
            print(f'Dataset {self.data_name} loaded from file')
    
    def load_point_data(self, data_dir, min_num_nodes, max_num_nodes, node_attributes, graph_labels):
        print('Loading point cloud dataset')
        name = 'FIRSTMM_DB'
        G = nx.Graph()
        # load data
        path = os.path.join(data_dir, name)
        data_adj = np.loadtxt(
            os.path.join(path, f'{name}_A.txt'), delimiter=',').astype(int)
        if node_attributes:
            data_node_att = np.loadtxt(os.path.join(path, f'{name}_node_attributes.txt'), 
                                    delimiter=',')
        data_node_label = np.loadtxt(os.path.join(path, f'{name}_node_labels.txt'), 
                                    delimiter=',').astype(int)
        data_graph_indicator = np.loadtxt(os.path.join(path, f'{name}_graph_indicator.txt'),
                                        delimiter=',').astype(int)
        if graph_labels:
            data_graph_labels = np.loadtxt(os.path.join(path, f'{name}_graph_labels.txt'), 
                                        delimiter=',').astype(int)

        data_tuple = list(map(tuple, data_adj))

        # add edges
        G.add_edges_from(data_tuple)
        # add node attributes
        for i in range(data_node_label.shape[0]):
            if node_attributes:
                G.add_node(i + 1, feature=data_node_att[i])
                G.add_node(i + 1, label=data_node_label[i])
        G.remove_nodes_from(list(nx.isolates(G)))

        # remove self-loop
        G.remove_edges_from(nx.selfloop_edges(G))

        # split into graphs
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        graphs = []
        max_nodes = 0
        for i in range(graph_num):
            # find the nodes for each graph
            nodes = node_list[data_graph_indicator == i + 1]
            G_sub = G.subgraph(nodes)
            if graph_labels:
                G_sub.graph['label'] = data_graph_labels[i]

            if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
                graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
                
        print('Loaded')
        return graphs

class EgoLargeDataset(SpectreGraphDataset):
    def __init__(self, data_file, adjs):
        self.data_name = 'ego'
        if data_file is None:
            self.adjs = adjs
        else:
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
            filename = os.path.join(base_path, data_file)
            _, _, G = self.load_ego_data(filename, dataset='citeseer')
            G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
            G = nx.convert_node_labels_to_integers(G)
            graphs = []
            for i in range(G.number_of_nodes()):
                G_ego = nx.ego_graph(G, i, radius=3)
                if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                    graphs.append(G_ego)
            self.adjs = [nx.adjacency_matrix(graph) for graph in graphs]
            print(f'Dataset {self.data_name} loaded from file')
    
    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    
    def load_ego_data(self, filename, dataset):
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            load = pickle.load(open(f"{filename}/ego/ind.{dataset}.{names[i]}", 'rb'), encoding='latin1')
            objects.append(load)
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file(f"{filename}/ego/ind.{dataset}.test.index")
        test_idx_range = np.sort(test_idx_reorder)

        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        features = vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        G = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(G)
        return adj, features, G

class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        super().__init__(cfg)
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs):
        test_len = int(round(len(graphs) * 0.2))
        train_len = int(round((len(graphs) - test_len) * 0.8))
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        super().prepare_data(datasets)


class GDSSGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        super().__init__(cfg)
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()
        
    def __getitem__(self, item):
        return self.inner[item]
        
    def prepare_data(self, graphs, data_name):
        dataset_ob = {'ego_small': EgoDataset, 'com_small': ComSmallDataset,
                        'grid': GridDataset, 'enz': EnzDataset, 'ego': EgoLargeDataset}.get(data_name)
        
        train_size = 0.7
        val_size = 0.1
        test_size = 0.2
        adjs = [torch.tensor(adj.todense()) for adj in graphs.adjs]
        train_val, test = train_test_split(adjs, train_size=train_size + val_size, shuffle=False)
        train, val = train_test_split(train_val, train_size=train_size / (train_size + val_size), random_state=42, shuffle=True)
        print(f'Dataset sizes: train {len(train)}, val {len(val)}, test {len(test)}')
        datasets = {'train': dataset_ob(data_file=None, adjs=train),
                    'val': dataset_ob(data_file=None, adjs=val), 
                    'test': dataset_ob(data_file=None, adjs=test)}
        return super().prepare_data(datasets)

class PointLobsterGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        super().__init__(cfg)
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()
        
    def __getitem__(self, item):
        return self.inner[item]
    
    def prepare_data(self, graphs, data_name):
        dataset_ob = {'point': PointDataset, 'lobster': LobsterDataset}.get(data_name)
    
        train_size = 0.7
        val_size = 0.2
        adjs = [torch.tensor(adj.todense()) for adj in graphs.adjs]
        data = adjs
        train = data[int(val_size*len(data)):int((train_size+val_size)*len(data))]
        val = data[:int(val_size*len(data))]
        test = data[int((train_size+val_size)*len(data)):]
        
        datasets = {'train': dataset_ob(data_file=None, adjs=train),
                    'val': dataset_ob(data_file=None, adjs=val), 
                    'test': dataset_ob(data_file=None, adjs=test)}
        
        print(f'Dataset sizes: train {len(train)}, val {len(val)}, test {len(test)}')
        return super().prepare_data(datasets)

class Comm20DataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = Comm20Dataset()
        return super().prepare_data(graphs)


class SBMDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = SBMDataset()
        return super().prepare_data(graphs)

class PlanarDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = PlanarDataset()
        return super().prepare_data(graphs)
    
class ProteinDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = ProteinsDataset('', None)
        return super().prepare_data(graphs)

class ComDataModule(GDSSGraphDataModule):
    def prepare_data(self):
        graphs = ComSmallDataset('', None)
        return super().prepare_data(graphs, data_name='com_small')
    
class EgoDataModule(GDSSGraphDataModule):
    def prepare_data(self):
        graphs = EgoDataset('', None)
        return super().prepare_data(graphs, data_name='ego_small')

class GridDataModule(GDSSGraphDataModule):
    def prepare_data(self):
        graphs = GridDataset('', None)
        return super().prepare_data(graphs, data_name='grid')

class EnzDataModule(GDSSGraphDataModule):
    def prepare_data(self):
        graphs = EnzDataset('', None)
        return super().prepare_data(graphs, data_name='enz')

class EgoLargeDataModule(GDSSGraphDataModule):
    def prepare_data(self):
        graphs = EgoLargeDataset('', None)
        return super().prepare_data(graphs, data_name='ego')

class PointDataModule(PointLobsterGraphDataModule):
    def prepare_data(self):
        graphs = PointDataset('', None)
        return super().prepare_data(graphs, data_name='point')
    
class LobsterDataModule(PointLobsterGraphDataModule):
    def prepare_data(self):
        graphs = LobsterDataset('', None)
        return super().prepare_data(graphs, data_name='lobster')

class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

