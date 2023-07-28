import argparse


def DPM_hyperparams(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=8, 
        help="number of jobs to run in parallel"
    )
    
    parser.add_argument(
        "--alternate",
        action="store_true"
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="cluster_net",
        help="choose a clustering method",
    )
    parser.add_argument(
        "--clusternet_hidden",
        type=int,
        default=40,
        help="The dimensions of the hidden dim of the clusternet. Defaults to 50.",
    )
    parser.add_argument(
        "--clusternet_hidden_layer_list",
        type=int,
        nargs="+",
        default=[40],
        help="The hidden layers in the clusternet. Defaults to [50, 50].",
    )
    parser.add_argument(
        "--transform_input_data",
        type=str,
        default="normalize",
        choices=["normalize", "min_max", "standard",
                "standard_normalize", "None", None],
        help="Use normalization for embedded data",
    )
    parser.add_argument(
        "--cluster_loss_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--init_cluster_net_weights",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--how_to_compute_mu",
        type=str,
        choices=["kmeans", "soft_assign"],
        default="soft_assign",
    )
    parser.add_argument(
        "--how_to_init_mu",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans",
    )
    parser.add_argument(
        "--how_to_init_mu_sub",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans_1d",
    )
    parser.add_argument(
        "--cluster_lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--subcluster_lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="ReduceOnP", choices=["StepLR", "None", "ReduceOnP"]
    )
    parser.add_argument(
        "--start_sub_clustering",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--subcluster_loss_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--start_splitting",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--subcluster_softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--split_prob",
        type=float,
        default=None,
        help="Split with this probability even if split rule is not met.  If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--merge_prob",
        type=float,
        default=None,
        help="merge with this probability even if merge rule is not met. If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--init_new_weights",
        type=str,
        default="same",
        choices=["same", "random", "subclusters"],
        help="How to create new weights after split. Same duplicates the old cluster's weights to the two new ones, random generate random weights and subclusters copies the weights from the subclustering net",
    )
    parser.add_argument(
        "--start_merging",
        type=int,
        default=15,
        help="The epoch in which to start consider merge proposals",
    )
    parser.add_argument(
        "--merge_init_weights_sub",
        type=str,
        default="highest_ll",
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_init_weights_sub",
        type=str,
        default="random",
        choices=["same_w_noise", "same", "random"],
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_merge_every_n_epochs",
        type=int,
        default=5,
        help="Example: if set to 10, split proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--raise_merge_proposals",
        type=str,
        default="brute_force_NN",
        help="how to raise merge proposals",
    )
    parser.add_argument(
        "--cov_const",
        type=float,
        default=0.005,
        help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const",
    )
    parser.add_argument(
        "--freeze_mus_submus_after_splitmerge",
        type=int,
        default=2,
        help="Numbers of epochs to freeze the mus and sub mus following a split or a merge step",
    )
    parser.add_argument(
        "--freeze_mus_after_init",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a new initialization",
    )
    parser.add_argument(
        "--use_priors",
        type=int,
        default=1,
        help="Whether to use priors when computing model's parameters",
    )
    parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
    parser.add_argument(
        "--pi_prior", type=str, default="uniform", choices=["uniform", None]
    )
    parser.add_argument(
        "--prior_dir_counts",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prior_kappa",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--prior_mu_0",
        type=str,
        default="data_mean",
    )
    parser.add_argument(
        "--prior_sigma_choice",
        type=str,
        default="data_std",
        choices=["iso_005", "iso_001", "iso_0001", "data_std"],
    )
    parser.add_argument(
        "--prior_sigma_scale",
        type=float,
        default=".005",
    )
    parser.add_argument(
        "--prior_sigma_scale_step",
        type=float,
        default=1.,
        help="add to change sigma scale between alternations"
    )
    parser.add_argument(
        "--compute_params_every",
        type=int,
        help="How frequently to compute the clustering params (mus, sub, pis)",
        default=1,
    )
    parser.add_argument(
        "--start_computing_params",
        type=int,
        help="When to start to compute the clustering params (mus, sub, pis)",
        default=5,
    )
    parser.add_argument(
        "--cluster_loss",
        type=str,
        help="What kind og loss to use",
        default="KL_GMM_2",
        choices=["diag_NIG", "isotropic", "isotropic_2",
                "isotropic_3", "isotropic_4", "KL_GMM_2"],
    )
    parser.add_argument(
        "--subcluster_loss",
        type=str,
        help="What kind og loss to use",
        default="isotropic",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )

    parser.add_argument(
        "--use_priors_for_net_params_init",
        type=bool,
        default=True,
        help="when the net is re-initialized after an AE round, if centers are given, if True it will initialize the covs and the pis using the priors, if false it will compute them using min dist."
    )
    # train params
    parser.add_argument(
        "--cluster_assignments",
        type=str,
        help="how to get the cluster assignment while training the AE, min_dist (hard assignment), forward_pass (soft assignment), pseudo_label (hard/soft assignment, TBD)",
        choices=["min_dist", "forward_pass", "pseudo_label"],
        default="forward_pass"
    )

    parser.add_argument(
        "--update_clusters_params",
        type=str,
        choices=["False", "only_centers",
                "all_params", "all_params_w_prior"],
        default="False",
        help="whether and how to update the clusters params (e.g., center) during the AE training"
    )
    parser.add_argument(
        "--init_cluster_net_using_centers",
        action="store_true"
    )
    parser.add_argument(
        "--reinit_net_at_alternation",
        action="store_true"
    )
    parser.add_argument(
        "--regularization",
        type=str,
        choices=["dist_loss", "cluster_loss"],
        help="which cluster regularization to use on the AE",
        default="cluster_loss"
    )
    return parser