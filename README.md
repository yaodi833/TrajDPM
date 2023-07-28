# TrajDPM
Code and data of the paper "Deep Dirichlet Process Mixture Model for Non-parametric Trajectory Clustering"

## dataset

We provide the Singapore dataset, which can be downloaded from the following [link](https://www.dropbox.com/sh/nrak8gsdzsbbvnk/AACeFW84RA0-yQjeVQiWvXu5a?dl=0).Please place the dataset in `./dataset/Singapore`.

## pretrain

```shell
python main.py --name Singapore  --mode pretrain --pretrain_epochs 10
```

The results of the pretrain will be placed in `./model/` (please create one manually if it doesn't exist)

## train

```shell
python main.py --name Singapore  --mode train --epochs 10 --alternate
```

The output of the trained encoder will be placed in `./embeddings/` (please create one manually if it doesn't exist)

