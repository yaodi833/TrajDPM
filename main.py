import torch
import argparse

from transformer import Transformer_DPM 
from train import train_model, pretrain
from DPM.argument import DPM_hyperparams

def minimal_hyperparams(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--dataset", type=str, required=True, help="datasets name")
    parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "train"], help="running mode(pretrain or train)")
    parser.add_argument("--gpu", type=int, default=0, help="gpu index")
    parser.add_argument("--logger", type=str, choices=["Tensorboard", "Dummylogger"],default="Tensorboard", help="logger to use")
    
    ###
    parser.add_argument("--origin_in", type=int, default=2, help="origin data dimension")
    parser.add_argument("--d_model", type=int, default=32, help="dimension of model")  
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers")
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of fcn")  
    parser.add_argument("--dropout", type=float, default=0.005, help="dropout")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight_decay")
    parser.add_argument("--activation", type=str, default="gelu", help="activation name")
    parser.add_argument("--ms_batch", type=int, default=5, help="The size of knn in meanshift loss")

    ###
    parser.add_argument("--temp", type=float, default=0.08, help="temperature for softmax.")
    parser.add_argument("--beta", type=float, default=1.0, help="coefficient of meanshift loss")
    parser.add_argument("--lam", type=float, default=0.8, help="hypermeter in clsloss")
    ###

    parser.add_argument("--train_batch_size", type=int, default=100, help="the batch_size of train")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="the batch_size of evaluation (validation or test)")
    parser.add_argument("--num_workers", type=int, default=15, help="the n_work parameter in dataloader")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="optimizer learning rate")
    parser.add_argument("--pretrain_epochs", type=int, default=30, help="the number of pretraining epochs")
    parser.add_argument("--epochs", type=int, default=40, help="the number of training epochs")

    parser.add_argument("--pretrain_model_path", type=str, default="./model/{0}_pretrain_model.pkl")
    parser.add_argument("--best_model_path", type=str, default="./model/{0}_best_model.pkl")
    parser.add_argument("--best_embeddings_path", type=str, default="./embeddings/{0}_best_embeddings.pkl")
    
    # cluster
    parser.add_argument("--init_k", default=8, type=int, help="number of initial clusters")
    parser.add_argument("--n_clusters", default=8, type=int, help="number of  clusters")
    parser.add_argument("--train_cluster_net", type=int, default=25, help="Number of epochs to pretrain the cluster net")
    parser.add_argument("--NIW_prior_nu", type=float, default=40, help="Need to be at least codes_dim + 1")    
    
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajDPM")
    parser = minimal_hyperparams(parser)
    parser = DPM_hyperparams(parser)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    model = Transformer_DPM(args, device)

    if args.mode == "pretrain":
        pretrain(model, args, device)
    elif args.mode == "train":
        train_model(model, args, device)
    else:
        raise NotImplementedError


    torch.cuda.empty_cache()


