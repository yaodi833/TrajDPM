import torch
import torch.nn as nn
import torch.nn.functional as F
from traj_transformer.traj_embedding import TrajEmbedding
from DPM.clusternet import ClusterNet
from data_loader import get_cluster_data_loader
import config

class Transformer_DPM(nn.Module):
    def __init__(self, args, device):
        super(Transformer_DPM, self).__init__()
  
        self.transformer = TrajEmbedding(fc_up_in_dim=args.origin_in, 
                              d_model=args.d_model, 
                              d_ff=args.d_ff, 
                              n_head=args.n_heads, 
                              num_encoder_layers=args.e_layers, 
                              max_length=config.max_length[args.name], 
                              device=device).to(device)
        self.clustering = ClusterNet(args, self.transformer)

        self.Sim = Similarity(temp=args.temp)

        self.train_data_loader = get_cluster_data_loader(args.name, args.eval_batch_size)
        self.val_data_loader = get_cluster_data_loader(args.name, args.eval_batch_size)


    def _init_clusters(self, centers=None):
        self.clustering.init_cluster(self.train_data_loader, self.val_data_loader, centers=centers)
        self.n_clusters = self.clustering.n_clusters
    
    def _update_clusters(self, latent_X, cluster_assign):
        if self.args.update_clusters_params == "only_centers":
            elem_count = cluster_assign.sum(axis=0)
            for k in range(self.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.clustering.update_cluster_center(latent_X.detach().cpu().numpy(), k, cluster_assign.detach().cpu().numpy())
    
    def _comp_clusters(self, args):
        used_centers_for_initialization = self.clustering.clusters if args.init_cluster_net_using_centers else None
        if args.reinit_net_at_alternation or args.clustering != "cluster_net":
            self._init_clusters(centers=used_centers_for_initialization)
        else:
            self.clustering.fit_cluster(self.train_data_loader, self.val_data_loader, centers=used_centers_for_initialization)
            self.n_clusters = self.clustering.n_clusters


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp

    def forward(self, x, y):
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        sim =  torch.einsum('bc,kc->bk', [x, y])
        return  sim / self.temp