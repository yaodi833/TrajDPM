#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch import optim
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure

from DPM.utils.training_utils import training_utils
from DPM.utils.priors import Priors
from DPM.utils.clustering_operations import init_mus_and_covs, compute_data_covs_hard_assignment
from DPM.utils.split_merge_operations import update_models_parameters_split, split_step, merge_step, update_models_parameters_merge
from DPM.Classifiers import MLP_Classifier, Subclustering_net

def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class ClusterNet(nn.Module):
    def __init__(self, args, feature_extractor):
        super(ClusterNet, self).__init__()

        self.args = args
        self.latent_dim = args.d_model
        self.n_clusters = args.n_clusters
        self.feature_extractor = feature_extractor
        self.n_jobs = args.n_jobs
        self.init_num = args.n_clusters
        self.device = torch.device(f"cuda:{args.gpu}")
        self.K = args.init_k

        self.centers = None
        self.mus = None
        self.covs = None
        self.pi = None
        self.mus_sub = None
        self.cov_sub = None
        self.pi_sub = None

        self.split_performed = False  
        self.merge_performed = False

        self.clusters = None
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        
        self.cluster_net = MLP_Classifier(args, k=self.K, codes_dim=self.latent_dim).to(self.device)

        self.training_utils = training_utils(args)

        self.subclustering_net = Subclustering_net(args, codes_dim=self.latent_dim, k=self.K).to(self.device)

        
        self.prior_sigma_scale = args.prior_sigma_scale
        if self.init_num > 0 and args.prior_sigma_scale_step != 0:
            self.prior_sigma_scale = args.prior_sigma_scale / (self.init_num * args.prior_sigma_scale_step)
        self.use_priors = args.use_priors
        self.prior = Priors(args, K=self.K, codes_dim=self.latent_dim, prior_sigma_scale=self.prior_sigma_scale) # we will use for split and merges even if use_priors is false

        self.mus_inds_to_merge = None
        self.mus_ind_to_split = None
        
    def forward(self, codes):
        return self.cluster_net(codes)        

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters)
        )
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, train_loader, val_loader, centers=None):
        """ Generate initial clusters using the clusternet
            init num is the number of time the clusternet was initialized (from the AE_ClusterPipeline module)
        """
        self.fit_cluster(train_loader, val_loader, centers)
        self.freeze()
        self.feature_extractor.unfreeze()
        self.feature_extractor.to(device=self.device)
        self.init_num += 1

    def fit_cluster(self, train_loader, val_loader, centers=None):
        self.centers = centers
        self.feature_extractor.freeze()
        self.unfreeze()
        self.fit(train_loader, val_loader, self.args.train_cluster_net)
        self.freeze()
        self.clusters = self.mus.cpu().numpy()
        self._set_K(self.K)
        self.feature_extractor.unfreeze()
        self.feature_extractor.to(device=self.device)

    def freeze(self) -> None:
        for param in self.cluster_net.parameters():
            param.requires_grad = False
        self.cluster_net.eval()
        for param in self.subclustering_net.parameters():
            param.requires_grad = False
        self.subclustering_net.eval()

    def unfreeze(self) -> None:
        for param in self.cluster_net.parameters():
            param.requires_grad = True
        self.cluster_net.train()
        for param in self.subclustering_net.parameters():
            param.requires_grad = True
        self.subclustering_net.train()

    def update_cluster_center(self, X, cluster_idx, assignments=None):
        """ Update clusters centers on a batch of data

        Args:
            X (torch.tensor): All the data points that were assigned to this cluster
            cluster_idx (int): The cluster index
            assignments: The probability of each cluster to be assigned to this cluster (would be a vector of ones for hard assignment)
        """
        n_samples = X.shape[0]
        for i in range(n_samples):
            if assignments[i, cluster_idx].item() > 0:
                self.count[cluster_idx] += assignments[i, cluster_idx].item()
                eta = 1.0 / self.count[cluster_idx]
                updated_cluster = (1 - eta) * self.clusters[cluster_idx] + eta * X[i] * assignments[i, cluster_idx].item()
                # updated_cluster = (1 - eta) * self.clusters[cluster_idx] + eta * X[i]
                self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X, how_to_assign="min_dist"):
        """ Assign samples in `X` to clusters """
        if how_to_assign == "min_dist":
            return self._update_assign_min_dist(X.detach().cpu().numpy())
        elif how_to_assign == "forward_pass":
            return self.get_model_resp(X)

    def _update_assign_min_dist(self, X):
        dis_mat = self._compute_dist(X)
        hard_assign = np.argmin(dis_mat, axis=1)
        return self._to_one_hot(torch.tensor(hard_assign))

    def _to_one_hot(self, hard_assignments):
        """
        Takes LongTensor with index values of shape (*) and
        returns a tensor of shape (*, num_classes) that have zeros everywhere
        except where the index of last dimension matches the corresponding value
        of the input tensor, in which case it will be 1.
        """
        return torch.nn.functional.one_hot(hard_assignments, num_classes=self.n_clusters)

    def _set_K(self, new_K):
        self.n_clusters = new_K
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate, pseudo-counts

    def get_model_params(self):
        mu, covs, pi, K = self.mus.cpu().numpy(), self.covs.cpu().numpy(), self.pi.cpu().numpy(), self.n_clusters
        return mu, covs, pi, K

    def get_model_resp(self, codes):
        self.cluster_net.to(device=self.device)
        if self.args.regularization == "cluster_loss":
            # cluster assignment should have grad
            logits = self.cluster_net(codes)
        else:
            # cluster assignment shouldn't have grad
            with torch.no_grad():
                logits = self.cluster_net(codes)
        return logits


    def fit(self, train_loader, val_loader, n_epochs):
        ## define the optimizer and learning rate scheduler
        cluster_params = torch.nn.ParameterList([p for n, p in self.cluster_net.named_parameters() if "class_fc2" not in n])
        cluster_net_opt = optim.Adam(cluster_params, lr=self.args.cluster_lr, weight_decay=self.args.weight_decay)
        cluster_net_opt.add_param_group({"params": self.cluster_net.class_fc2.parameters()})
        sub_clus_opt = optim.Adam(self.subclustering_net.parameters(), lr=self.args.subcluster_lr, weight_decay=self.args.weight_decay)
        
        if self.args.lr_scheduler == "StepLR":
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_net_opt, step_size=20)
        elif self.args.lr_scheduler == "ReduceOnP":
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cluster_net_opt, mode="min", factor=0.5, patience=4)
        else:
            raise NotImplementedError
        
        self.freeze_mus_after_init_until = 0
        for epoch in range(n_epochs):

            ## train
            self.cluster_net.train()
            self.subclustering_net.train()

            self.split_performed = False
            self.merge_performed = False

            codes = []
            train_gt = []
            train_resp = []
            train_resp_sub = []


            for traj, traj_mask, traj_length, label in train_loader:

                traj, traj_mask, traj_length = traj.to(self.device), traj_mask.to(self.device), traj_length.to(self.device)

                ## first stage: clusternet (EM process)
                cluster_net_opt.zero_grad()

                embeddings = self.feature_extractor(traj, traj_mask, traj_length)
                embeddings = embeddings.view(-1, self.latent_dim)
                
                logits = self.cluster_net(embeddings)
                ### only gather codes
                if epoch != 0:
                    cluster_loss = self.training_utils.cluster_loss_function(embeddings, logits, self.mus, self.K, self.covs, self.pi)
                    cluster_loss = self.args.cluster_loss_weight * cluster_loss

                    cluster_loss.backward()

                    cluster_net_opt.step()

                ## second stage: subclusternet 
                ###  annotation: this stage will begin after serveral epochs
                if self.args.start_sub_clustering <= epoch:
                    sub_clus_opt.zero_grad()

                    embeddings = self.feature_extractor(traj, traj_mask, traj_length)
                    embeddings = embeddings.view(-1, self.latent_dim)
                    logits = self.cluster_net(embeddings).detach()

                    sublogits = self.subcluster(embeddings, logits)
                    ### only gather codes
                    if epoch != 0:
                        subcluster_loss = self.training_utils.subcluster_loss_function(embeddings, logits, sublogits, self.K, self.mus_sub, self.covs_sub, self.pi_sub)
                        subcluster_loss = self.args.subcluster_loss_weight * subcluster_loss

                        subcluster_loss.backward()
                        sub_clus_opt.step()
                    
                    train_resp_sub.append(sublogits.detach())


                
                codes.append(embeddings.detach())
                train_gt.append(label.detach())
                train_resp.append(logits.detach())

            codes = torch.cat(codes, dim=0).cpu()
            train_gt = torch.cat(train_gt, dim=0).cpu()
            train_resp = torch.cat(train_resp, dim=0).cpu()
            if epoch >= self.args.start_sub_clustering:
                train_resp_sub = torch.cat(train_resp_sub, dim=0).cpu()
            

            if epoch == 0:
                self.prior.init_priors(codes)
                if self.centers is not None:
                    self.mus = torch.from_numpy(self.centers).cpu()
                    self.centers = None
                    self.init_covs_and_pis_given_mus()
                    self.freeze_mus_after_init_until = epoch + self.args.freeze_mus_after_init
                else:
                    self.freeze_mus_after_init_until = 0
                    self.mus, self.covs, self.pi, _ = init_mus_and_covs(codes, self.K, self.args.how_to_init_mu, train_resp, self.args.use_priors, self.prior, 0, self.device)
            else:   
                # Compute mus and perform splits/merges
                perform_split = self.training_utils.should_perform_split(epoch) and self.centers is None
                perform_merge = self.training_utils.should_perform_merge(epoch, self.split_performed) and self.centers is None
                # do not compute the mus in the epoch(s) following a split or a merge
                if self.centers is not None:
                    # we have initialization from somewhere
                    self.mus = torch.from_numpy(self.centers).cpu()
                    self.centers = None
                    self.init_covs_and_pis_given_mus()
                    self.freeze_mus_after_init_until = epoch + self.args.freeze_mus_after_init
                
                freeze_mus = self.training_utils.freeze_mus(epoch, self.split_performed) or epoch <= self.freeze_mus_after_init_until
                if not freeze_mus:
                    self.pi, self.mus, self.covs = self.training_utils.comp_cluster_params(train_resp, codes, self.pi, self.K, self.prior)
                
                # update the paras in subcluster net
                if self.args.start_sub_clustering == epoch + 1:
                    self.pi_sub, self.mus_sub, self.covs_sub = self.training_utils.init_subcluster_params(train_resp, train_resp_sub, codes, self.K, self.prior)                      
                elif self.args.start_sub_clustering <= epoch and not freeze_mus:
                    self.pi_sub, self. mus_sub, self.covs_sub = self.training_utils.comp_subcluster_params(train_resp, train_resp_sub, codes, self.K, self.mus_sub, self.covs_sub, self.pi_sub, self.prior,)

                # split
                if perform_split and not freeze_mus:
                    self.training_utils.last_performed = "split"
                    split_decisions = split_step(self.K, codes, train_resp, train_resp_sub, self.mus, self.mus_sub, self.args.cov_const, self.args.alpha, self.args.split_prob, self.prior)
                    if split_decisions.any():
                        self.split_performed = True
                        cluster_net_opt, sub_clus_opt = self.perform_split(split_decisions, [cluster_net_opt, sub_clus_opt], codes, train_resp, train_resp_sub)

                # merge
                if perform_merge and not freeze_mus:
                    self.training_utils.last_performed = "merge"
                    mus_to_merge, highest_ll_mus = merge_step(self.mus, train_resp, codes, self.K, self.args.raise_merge_proposals, self.args.cov_const, self.args.alpha, self.args.merge_prob, prior=self.prior)
                    if len(mus_to_merge) > 0:
                        # there are mus to merge
                        self.merge_performed = True
                        cluster_net_opt, sub_clus_opt = self.perform_merge(mus_to_merge, highest_ll_mus, [cluster_net_opt, sub_clus_opt], codes, train_resp)

            if self.split_performed or self.merge_performed:
                self.update_params_split_merge()
                print("Current number of clusters: ", self.K)

            ## validation
            self.cluster_net.eval()
            self.subclustering_net.eval()
            codes = []
            val_resp = []
            val_gt = []

            loss = 0

            for traj, traj_mask, traj_length, label in val_loader:
                traj, traj_mask, traj_length = traj.to(self.device), traj_mask.to(self.device), traj_length.to(self.device)

                with torch.no_grad():
                    embeddings = self.feature_extractor(traj, traj_mask, traj_length).view(-1, self.latent_dim)
                    logits = self.cluster_net(embeddings)

                cluster_loss = self.training_utils.cluster_loss_function(embeddings, logits, self.mus, self.K, self.covs, self.pi)
                loss += self.args.cluster_loss_weight * cluster_loss

                if self.args.start_sub_clustering <= epoch:
                    with torch.no_grad():
                        sublogits = self.subcluster(embeddings, logits)

                    subcluster_loss = self.training_utils.subcluster_loss_function(embeddings, logits, sublogits, self.K, self.mus_sub, self.covs_sub, self.pi_sub)
                    loss += self.args.subcluster_loss_weight * subcluster_loss

                val_gt.append(label)
                val_resp.append(logits)

            val_gt = torch.cat(val_gt, dim=0)
            val_resp = torch.cat(val_resp, dim=0)

            z = val_resp.argmax(axis=1).cpu()

            nmi = normalized_mutual_info_score(val_gt, z)
            ari = adjusted_mutual_info_score(val_gt, z)
            acc = self.training_utils.cluster_acc(val_gt, z)
            
            print("DPM: epoch{0} \tacc:{1:4f}\tnmi:{2:4f}\tari:{3:4f}\tloss:{4:4f}".format(
                epoch, acc, nmi, ari, loss
            ))
            ## update learning rate scheduler
            cluster_scheduler.step(loss)

    def subcluster(self, codes, logits):
        # cluster codes into subclusters
        sub_clus_resp = self.subclustering_net(codes)  # unnormalized
        z = logits.argmax(-1)

        # zero out irrelevant subclusters
        mask = torch.zeros_like(sub_clus_resp)
        mask[np.arange(len(z)), 2 * z] = 1.
        mask[np.arange(len(z)), 2 * z + 1] = 1.

        # perform softmax
        sub_clus_resp = torch.nn.functional.softmax(sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm, dim=1)
        return sub_clus_resp

    def perform_split(self, split_decisions, optim, codes, train_resp, train_resp_sub,):
        # split_decisions is a list of k boolean indicators of whether we would want to split cluster k
        # update the cluster net to have the new K
        clus_opt, subclus_opt = optim

        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)

        self.cluster_net.update_K_split(split_decisions, self.args.init_new_weights, self.subclustering_net)
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self.device)

        mus_ind_to_split = torch.nonzero(split_decisions.clone().detach(), as_tuple=False)
        
        self.mus_new, self.covs_new, self.pi_new, self.mus_sub_new, self.covs_sub_new, self.pi_sub_new = update_models_parameters_split(
            split_decisions, self.mus, self.covs, self.pi, mus_ind_to_split, self.mus_sub, self.covs_sub, self.pi_sub, codes, train_resp,
            train_resp_sub, self.args.how_to_init_mu_sub, self.prior, self.args.use_priors)
        
        # update K
        print(f"Splitting clusters {np.arange(self.K)[split_decisions.bool().tolist()]}")
        self.K += len(mus_ind_to_split)

        subclus_opt = self.update_subcluster_net_split(split_decisions, subclus_opt)

        self.mus_ind_to_split = mus_ind_to_split

        return clus_opt, subclus_opt

    def update_subcluster_net_split(self, split_decisions, optim):
        # update the subcluster net to have the new K
        subclus_opt = optim

        # remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)

        self.subclustering_net.update_K_split(split_decisions, self.args.split_init_weights_sub)

        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())
        
        return subclus_opt

    def perform_merge(self, mus_lists_to_merge, highest_ll_mus, optim, codes, train_resp, use_priors=True):
        clus_opt, sub_clus_opt = optim
        
        print(f"Merging clusters {mus_lists_to_merge}")

        mus_lists_to_merge = torch.tensor(mus_lists_to_merge)
        inds_to_mask = torch.zeros(self.K, dtype=bool)
        inds_to_mask[mus_lists_to_merge.flatten()] = 1

        self.mus_new, self.covs_new, self.pi_new, self.mus_sub_new, self.covs_sub_new, self.pi_sub_new,= update_models_parameters_merge(
            mus_lists_to_merge, inds_to_mask, self.K, self.mus, self.covs, self.pi, self.mus_sub, self.covs_sub, self.pi_sub, 
            codes, train_resp, self.prior, use_priors=self.args.use_priors, how_to_init_mu_sub=self.args.how_to_init_mu_sub)
        # adjust k
        self.K -= len(highest_ll_mus)

        # update the subclustering net
        sub_clus_opt = self.update_subcluster_nets_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus, sub_clus_opt)


        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        
        # update cluster net
        self.cluster_net.update_K_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus, init_new_weights=self.args.init_new_weights)
        # add parameters to the optimizer
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())

        self.cluster_net.class_fc2.to(self.device)
        self.mus_inds_to_merge = mus_lists_to_merge

        return clus_opt, sub_clus_opt

    def update_subcluster_nets_merge(self, merge_decisions, pairs_to_merge, highest_ll, optim):
        # update the cluster net to have the new K
        subclus_opt = optim
        
        # remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_merge(merge_decisions, pairs_to_merge=pairs_to_merge, highest_ll=highest_ll, init_new_weights=self.args.merge_init_weights_sub)        
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())

        return subclus_opt

    def update_params_split_merge(self):
        self.mus = self.mus_new
        self.covs = self.covs_new
        self.mus_sub = self.mus_sub_new
        self.covs_sub = self.covs_sub_new
        self.pi = self.pi_new
        self.pi_sub = self.pi_sub_new

    def init_covs_and_pis_given_mus(self):
        # each point will be hard assigned to its closest cluster and then compute covs and pis.
        # compute dist mat
        if self.args.use_priors_for_net_params_init:
            _, cov_prior = self.prior.init_priors(self.mus)  # giving mus and nopt codes because we only need the dim for the covs
            self.covs = torch.stack([cov_prior for k in range(self.K)])
            p_counts = torch.ones(self.K) * 10
            self.pi = p_counts / float(self.K * 10)  # a uniform pi prior

        else:
            dis_mat = torch.empty((len(self.codes), self.K))
            for i in range(self.K):
                dis_mat[:, i] = torch.sqrt(((self.codes - self.mus[i]) ** 2).sum(axis=1))
            # get hard assingment
            hard_assign = torch.argmin(dis_mat, dim=1)

            # data params
            vals, counts = torch.unique(hard_assign, return_counts=True)
            if len(counts) < self.K:
                new_counts = []
                for k in range(self.K):
                    if k in vals:
                        new_counts.append(counts[vals == k])
                    else:
                        new_counts.append(0)
                counts = torch.tensor(new_counts)
            pi = counts / float(len(self.codes))
            data_covs = compute_data_covs_hard_assignment(hard_assign.numpy(), self.codes, self.K, self.mus.cpu(), self.prior)
            if self.use_priors:
                covs = []
                for k in range(self.K):
                    codes_k = self.codes[hard_assign == k]
                    cov_k = self.prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])
                    covs.append(cov_k)
                covs = torch.stack(covs)
            self.covs = covs
            self.pi = pi


