import os.path as osp
import os
from src.flow_utils.graphutils import *
from src.flow_utils.flow import *
from src.flow_utils._aux import *
import torch
import pickle
from torch_geometric.data import Dataset, download_url
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class TaxiDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = "power"
        self.split = split
        self.num_graphs = 200
        #super().__init__(root, transform, pre_transform, pre_filter)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def process(self):
        # Reading taxi data

        # Train-test-val
           
        #Preprocessing



        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 1, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        
        
        torch.save(self.collate(data_list), self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
class TaxiGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': PowerDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': PowerDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': PowerDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
