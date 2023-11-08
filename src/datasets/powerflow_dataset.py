import os.path as osp
from src.flow_utils import *
import torch
import pickle
from torch_geometric.data import Dataset, download_url
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class PowerDataset(Dataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = "power"
        self.split = split
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def process(self):
        # Reading power data
        data_folder = "../../data/"
        G = read_net(data_folder + self.dataset_name + "_net.csv")

        pfile = open(data_folder + "features_" + self.dataset_name + ".pkl", 'rb')
        features = pickle.load(pfile)
        pfile.close()

        pfile = open(data_folder + "flows_" + self.dataset_name + ".pkl", 'rb')
        flows = pickle.load(pfile)
        pfile.close()

        #Preprocessing
        G, flows, features = make_non_neg_norm(G, flows, features) #Converts flow estimation instance to a non-negative one, i.e. where every flow is non-negative.
        features = normalize_features(features) #Normalize features to 0-1
        print("erenpolat")
        print(G.shape)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
class PowerGraphDataModule(AbstractDataModule):
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
    
pd = PowerDataset(dataset_name="power", split="train", root="../../data")
pd.process()