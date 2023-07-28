from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch
import config

from tools import *

class PretrainDataset(Dataset):
    def __init__(self, data, phase, data_features, ns_matrix_path):
        super(PretrainDataset).__init__()

        (lon_mean, lon_std), (lat_mean, lat_std) = data_features
        self.data = []
        self.phase = phase
        

        self.cell_matrix = []  
        with open(ns_matrix_path[0], "rb")as f:
            self.cell_matrix.append(pickle.load(f))
        with open(ns_matrix_path[1], "rb")as f:
            self.cell_matrix.append(pickle.load(f))
        self.dis_matrix = None


        for i in range(len(data)):
            traj = torch.tensor(data[i])
            traj = traj - torch.tensor([lon_mean, lat_mean])
            traj = traj * torch.tensor([1 / lon_std, 1 / lat_std])
            self.data.append(traj.float())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.phase == "train":
            # traj_list = [anchor, positive, negative]
            traj_list = [self.data[index]]

            traj_list.append(_get_positive_traj(self.data[index]))
            
            negative_id = _get_negative_traj(index, cell_matrix=self.cell_matrix)

            traj_list.append(self.data[negative_id])
        elif self.phase == "embed" or self.phase == "meanshift":
            traj_list = [self.data[index]]
        return traj_list

def _get_negative_traj(index, cell_matrix=None):
    relation_matrix1 = cell_matrix[0][index]
    relation_matrix2 = cell_matrix[1][index]
    if (~relation_matrix1 & ~relation_matrix2).any(): 
        calidate = np.flatnonzero(~relation_matrix1 & ~relation_matrix2)
    else:
        calidate = np.flatnonzero(~relation_matrix1 | ~relation_matrix2)        
    return np.random.choice(calidate, 1)[0]


def _get_positive_traj(traj):

    if len(traj) < 10:
        p = traj
    else:
        a1, a3, a5 = 0, len(traj)//2, len(traj)
        a2, a4 = (a1 + a3)//2, (a3 + a5)//2

        if np.random.rand() > 0.5:
            sub_traj = traj[a2:a5]
        else:
            sub_traj = traj[a1:a4]
        idx = np.random.rand(len(sub_traj)) < 0.6
        idx[0], idx[-1] = True, True
        p = sub_traj[idx]
    
    return p

def _pad_traj(traj, max_length):
    _, D = traj.shape
    padding_traj = torch.zeros((max_length - len(traj), D))
    traj = torch.vstack((traj, padding_traj))
    return traj.numpy().tolist()


def _generate_square_subsequent_mask(traj_num, max_length, lenths):
    """
    padding_mask
    lenths: [lenth1,lenth2...]
    """
    mask = torch.ones(traj_num, max_length) == 1  # mask batch_size*3 x max_lenth

    for i, this_lenth in enumerate(lenths):
        for j in range(this_lenth):
            mask[i][j] = False

    return mask


def _get_standard_inputs(inputs):
    max_length = 0
    traj_tensor = []
    traj_length = []

    for b in inputs:
        for t in b:
            traj_length.append(len(t))
            if len(t) > max_length:
                max_length = len(t)

    for b_trajs in inputs:
        temp_b_trajs = []
        for traj in b_trajs:
            temp_b_trajs.append(_pad_traj(traj, max_length))
        traj_tensor.append(temp_b_trajs)

    traj_tensor = torch.tensor(traj_tensor, dtype=torch.float)  # traj_tensor [N, 3, S, 2]

    padding_mask = _generate_square_subsequent_mask(len(traj_length), max_length, traj_length)  # mask batch_size*3 x max_lenth

    return traj_tensor, padding_mask, traj_length


def _my_collect_function(batch):

    traj_list_all, padding_mask, traj_lengths = _get_standard_inputs(batch)  # inputs [batch_size, 3, sequence_len, embedding_size]  padding_mask=[traj_num, sequence_len]

    return traj_list_all, padding_mask, traj_lengths

    
def get_encoder_data_loader(dataset_name, phase, train_batch_size, eval_batch_size, num_workers):
    traj_path = config.traj_path[dataset_name]
    data_features = config.data_feature[dataset_name]
    ns_matrix_path = config.negativesample[dataset_name]

    traj = pload(traj_path)

    if phase == "train":
        is_shuffle = True
        batch_size = train_batch_size
    elif phase == "embed":
        is_shuffle = False
        batch_size = eval_batch_size
    elif phase == "meanshift":
        # sample 1000 traj
        is_shuffle = True
        batch_size = train_batch_size
        if len(traj) >= 1000:
            sample_index = np.random.randint(0, len(traj) - 999)
            traj = traj[sample_index: sample_index + 1000]

    dataset = PretrainDataset(traj, phase, data_features, ns_matrix_path)


    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, collate_fn=_my_collect_function)

    return data_loader


def get_cluster_data_loader(dataset_name, batch_size):
    traj_path = config.traj_path[dataset_name]
    label_path = config.label_path[dataset_name]
    data_features = config.data_feature[dataset_name]


    trajs = pload(traj_path)

    (lon_mean, lon_std), (lat_mean, lat_std) = data_features
    traj_nom = []
    for i in range(len(trajs)):
        traj = torch.tensor(trajs[i])
        traj = traj - torch.tensor([lon_mean, lat_mean])
        traj = traj * torch.tensor([1 / lon_std, 1 / lat_std])
        traj_nom.append([traj.float()])
    
    # traj_nom: data_size x 1 x max_length x origin_in
    traj_tensor, padding_mask, traj_length = _get_standard_inputs(traj_nom)
    traj_length = torch.LongTensor(traj_length)

    label = pload(label_path)
    label = torch.Tensor(label).to(torch.float32)

    return DataLoader(TensorDataset(traj_tensor, padding_mask, traj_length,  label), batch_size, num_workers=16)


class FinetuneDataset(Dataset):
    def __init__(self, data, data_features, labels, assignments):
        super(FinetuneDataset).__init__()

        (lon_mean, lon_std), (lat_mean, lat_std) = data_features
        self.data = []
        self.assignments = assignments
        self.labels = labels
        

        for i in range(len(data)):
            traj = torch.tensor(data[i])
            traj = traj - torch.tensor([lon_mean, lat_mean])
            traj = traj * torch.tensor([1 / lon_std, 1 / lat_std])
            self.data.append(traj.float())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # traj_list = [anchor, positive, negative]
        traj_list = [self.data[index]]
        positive_id = np.random.choice(np.flatnonzero(self.assignments == self.assignments[index]), 1)[0]
        traj_list.append(self.data[positive_id])
        negative_id = np.random.choice(np.flatnonzero(self.assignments != self.assignments[index]), 1)[0]
        traj_list.append(self.data[negative_id])

        return traj_list

def get_finetune_data_loader(dataset_name, train_batch_size, num_workers, assignments):
    traj_path = config.traj_path[dataset_name]
    data_features = config.data_feature[dataset_name]
    label_path = config.label_path[dataset_name]
    traj = pload(traj_path)
    labels = pload(label_path)
    dataset = FinetuneDataset(traj, data_features, labels, assignments)

    data_loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=num_workers, collate_fn=_my_collect_function)

    return data_loader

