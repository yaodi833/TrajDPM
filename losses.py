import torch
import torch.nn as nn
import h5py
import os

def meanshiftLoss(embeddings, ms_batch):
    embeddings = nn.functional.normalize(embeddings, dim=1)
    dist_embeddings =  2 - 2 * torch.einsum('bc,kc->bk', [embeddings, embeddings])
    # select the k nearest neighbors
    _, nn_index = dist_embeddings.topk(ms_batch, dim=1, largest=False)
    nn_dist_q = torch.gather(dist_embeddings, 1, nn_index)
    
    meanshift_loss = (nn_dist_q.sum(dim=1) / ms_batch).mean()
    return meanshift_loss

def clusterLoss(args, latent_X, cluster_assignment, clustering):
    if args.regularization == "dist_loss":
            dist_loss = torch.tensor(0.0).cuda()
            clusters = torch.FloatTensor(clustering.clusters).cuda()
            for i in range(args.eval_batch_size):
                diff_vec = latent_X[i] - clusters[cluster_assignment.argmax(-1)[i]]
                sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
                dist_loss += 0.5 * args.beta * torch.squeeze(sample_dist_loss)
            reg_loss = dist_loss

    elif args.regularization == "cluster_loss":
            # const clustering variables (w.r.t this training stage)
            mus, covs, pi, K = clustering.get_model_params()

            reg_loss = clustering.training_utils.cluster_loss_function(
                latent_X.detach(),
                cluster_assignment,
                model_mus=torch.from_numpy(mus),
                K=K,
                model_covs=torch.from_numpy(covs) if args.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                pi=torch.from_numpy(pi)
            )
    return reg_loss

def clsLoss(args, sim_p,sim_n):
    
    delta = torch.exp(sim_p) + args.lam * torch.exp(sim_n)
    # delta = torch.exp(sim_n)
    delta = torch.sum(delta, dim=-1)
    # print(torch.exp(sim_p))
    # print(torch.exp(sim_n))
    # print(delta)
    # print((torch.exp(torch.diag(sim_p)) / delta))
    return -torch.log(torch.exp(torch.diag(sim_p)) / delta).mean()