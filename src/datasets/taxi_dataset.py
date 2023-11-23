import os
import pathlib

import torch
import torch.nn.functional as F 
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url, Data

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class TaxiDataset(InMemoryDataset):
    def __init__(self, split, root, dataset_name, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 182
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        pyg_graphs = []
        print("BURDA HACI")
        for idx in range(self.num_graphs):
            tensor = torch.load(f"{self.root}/tensor{idx}.pt")
            #print(f"{root}/tensor{idx}.pt")
            pyg_graphs.append(tensor)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(pyg_graphs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])


    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        data_list = torch.load(self.raw_paths[file_idx[self.split]])
        max_flow = 30
        #we can add preprocessing here
        for idx, graph in enumerate(data_list):
            temp_edge_attr = graph.edge_attr
            edge_attr = F.one_hot(temp_edge_attr.long(), max_flow + 2).to(torch.float)
            node_attr = torch.ones(graph.x.shape[0], 1)
            data_list[idx].edge_attr = edge_attr
            data_list[idx].x = node_attr
            data_list[idx].n_nodes = data_list[idx].x.shape[0].unsqueeze()
            data_list[idx].num_nodes = data_list[idx].x.shape[0]
        torch.save(self.collate(data_list), self.processed_paths[0])

class TaxiDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        print("Rootpath", root_path)
        print("datadir", self.datadir)

        datasets = {'train': TaxiDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': TaxiDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': TaxiDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class TaxiDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'taxi_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

