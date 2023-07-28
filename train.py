import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans

import numpy as np
import time
import datetime
import config
from transformer import Transformer_DPM
from data_loader import get_encoder_data_loader, get_finetune_data_loader
from losses import meanshiftLoss, clsLoss
from metrics import cluster_acc, nmi_score, ari_score
from tools import pload, pdump, Dummylogger

def pretrain(model: Transformer_DPM, args, device):

    print("Start pretraining ... ")

    train_loader = get_encoder_data_loader(args.name, "train", args.train_batch_size, args.eval_batch_size, args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    construction_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    construction_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    time_now = time.time()

    
    for epoch in range(args.pretrain_epochs):
        
        model.train()

        epoch_begin_time = time.time()
        construction_loss_all = 0.0

        # Stage one: construction learning
        for trajs, traj_masks, traj_lengths in train_loader:
            construction_optimizer.zero_grad()
            trajs, traj_masks = trajs.to(device), traj_masks.to(device)

            with torch.set_grad_enabled(True):
                vectors = model.transformer(trajs, traj_masks, traj_lengths)  # vecters [B,N_S,D]
            construction_loss = construction_criterion(vectors[:, 0, :], vectors[:, 1, :], vectors[:, 2, :])

            construction_loss.backward()
            construction_optimizer.step()

            construction_loss_all += construction_loss.detach()
        
        
        # Stage two: meanshift
        optimizer.zero_grad()
        embeddings = get_embed(model, args, device, "meanshift")
        meanshift_loss = meanshiftLoss(embeddings, args.ms_batch)
       
        meanshift_loss.backward()
        optimizer.step()
        
        epoch_end_time = time.time()
        print("\nEpoch {}/{}:\nConstruction Loss: {:.4f}\tMeanShift Loss: {:.4f}\tTime: {} m {} s".format(
            epoch+1, args.pretrain_epochs, construction_loss_all, meanshift_loss, 
            (epoch_end_time - epoch_begin_time) // 60, int((epoch_end_time - epoch_begin_time) % 60)))        
    
        torch.save(model.transformer.state_dict(), args.pretrain_model_path.format(args.name))

    time_end = time.time()

    print("\nPretraining complete in {:.0f}m {:.0f}s".format((time_end - time_now) // 60, (time_end - time_now) % 60))     


def train_model(model: Transformer_DPM, args, device):
    
    model.transformer.load_state_dict(torch.load(args.pretrain_model_path.format(args.name)))

    # init logger
    if args.logger == "Tensorboard":
        timestr = "%m-%d_%H:%M"
        log_writer = SummaryWriter(f"./log/{args.name}/TensorBoard_{datetime.datetime.today().strftime(timestr)}/")
    elif args.logger == "Dummylogger":
        # use dummy logger during debug
        log_writer = Dummylogger()
    else:
        raise NotImplementedError

    # model._init_clusters()
    # get train data

    time_now = time.time()

    best_performance = [-1.0, -1.0, -1.0]
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=1e-5, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
  
        print("-"*7 + "epoch " + str(epoch)+ "-"*7)
        # Stage one : train the cluster net
        model.train()
        model._comp_clusters(args)   

        # Stage two : train the encoder
        model.train()
        with torch.no_grad():
            embeddings = get_embed(model, args, device, "embed")
            assignments = model.clustering.update_assign(embeddings, args.cluster_assignments).argmax(axis=1).cpu().numpy()
        
        finetune_dataloader = get_finetune_data_loader(args.name, args.train_batch_size, args.num_workers, assignments)
        for trajs, traj_masks, traj_lengths in finetune_dataloader:
            optimizer.zero_grad()

            trajs, traj_masks = trajs.to(device), traj_masks.to(device)

            vectors = model.transformer(trajs, traj_masks, traj_lengths) # vecters [B,3,D]

            sim_p = model.Sim(vectors[:, 0, :], vectors[:, 1, :])
            sim_n = model.Sim(vectors[:, 0, :], vectors[:, 2, :])

            loss = clsLoss(args, sim_p, sim_n)
            loss.backward()

            optimizer.step()

        # validation
        model.eval()
        val_begin_time = time.time()

        with torch.no_grad():
            embeddings = get_embed(model, args, device, "embed")
            assignments = model.clustering.update_assign(embeddings, args.cluster_assignments)
        pred = assignments.argmax(axis=1).cpu().numpy()
        ground_truth = pload(config.label_path[args.name])
        acc = np.round(cluster_acc(ground_truth, pred), 5)
        nmi = np.round(nmi_score(ground_truth, pred),5)
        ari = np.round(ari_score(ground_truth, pred), 5)

        val_end_time = time.time()

        log_writer.add_scalar(f"Transformer/acc", acc, epoch)
        log_writer.add_scalar(f"Transformer/nmi", nmi, epoch)
        log_writer.add_scalar(f"Transformer/ari", ari, epoch)

        print("\tacc: {:.4f}\tnmi: {:.4f}\tari: {:.4f}\tTime: {} m {} s".format(
            acc, nmi, ari, (val_end_time -val_begin_time) // 60, 
            int((val_end_time -val_begin_time) % 60)
        ))

        pred2 = KMeans(n_clusters=args.n_clusters, n_init='auto').fit_predict(embeddings.cpu().numpy())
        acc2 = np.round(cluster_acc(ground_truth, pred2), 5)
        nmi2 = np.round(nmi_score(ground_truth, pred2),5)
        ari2 = np.round(ari_score(ground_truth, pred2), 5)
        print("Kmeans:\tacc: {:.4f}\tnmi: {:.4f}\tari: {:.4f}".format(acc2, nmi2, ari2))

        if best_performance[0] < acc:
            best_performance[0] = acc
            best_performance[1] = nmi
            best_performance[2] = ari
            torch.save(model.state_dict(), args.best_model_path.format(args.name))
            pdump(embeddings.cpu().detach().numpy(), args.best_embeddings_path.format(args.name))

    time_end = time.time()

    print("\nAll training complete in {:.0f}m {:.0f}s".format((time_end - time_now) // 60, (time_end - time_now) % 60))
    print(f"Best performance::  acc: {best_performance[0]:.4f}  nmi: {best_performance[1]:.4f}  ari:{best_performance[2]:.4f}")



def get_embed(model, args, device, phase):
    data_loader = get_encoder_data_loader(args.name, phase, args.train_batch_size, args.eval_batch_size, args.num_workers)
    all_vectors = []

    for trajs, traj_masks, traj_lengths in data_loader:
        trajs, traj_masks = trajs.to(device), traj_masks.to(device)
        vectors = model.transformer(trajs, traj_masks, traj_lengths)  # vecters [B,N_S,D]

        all_vectors.append(vectors.squeeze(1))

    all_vectors = torch.cat(all_vectors, dim=0)

    return all_vectors