# MoCL: Contrastive Learning on Molecular Graphs with Multi-level Domain Knowledge

This is the official code repository of MoCL.  

## Paper Abstract
In this paper, we study graph contrastive learning in the context of biomedical domain, where molecular graphs are present. We propose a novel framework called MoCL, which utilizes domain knowledge at both local-and global-level to assist representation learning. The local-level domain knowledge guides the augmentation process such that variation is introduced without changing graph semantics. The global-level knowledge encodes the similarity information between graphsin the entire dataset and helps to learn representations with richer semantics. The entire model is learned through a double contrastobjective. We evaluate MoCL on various molecular datasets under both linear and semi-supervised settings and results show that MoCL achieves state-of-the-art performance.

The current version of the paper can be found [HERE](https://arxiv.org/abs/2106.04509).

## Installation
The code are tested on python 3.7 and following are the dependencies:

```
pytorch = 1.6.0
cuda = 10.2
torch-geometric = 1.6.3 (need to specify torch and cuda version during installation)
rdkit = 2020.09.1.0
tqdm = 4.57.0
tensorboardx = 2.1
```

## Key files
[`isostere_transformations_new.json`](https://github.com/illidanlab/MoCL-DK/blob/master/isostere_transformations_new.json): proposed augmentation rules derived from domain knowledge (written in SMARTS format). 

> SMARTS is a language that allows you to specify substructures using rules that are straightforward extensions of SMILES. For example, to search a database for phenol-containing structures, one would use the SMARTS string [OH]c1ccccc1, which should be familiar to those acquainted with SMILES.

More details can be found [here](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html). 

## Prepare rule_indicator matrix and sim_matrix_nb_{neighbor_size} matrix
rule_indicator_new.pkl contains a matrix M (n_mols x n_rules) derived from domain rules and will be used in domain augmentation. M[i, j] = k indicates i-th molecule has k substructures matched for j-th rule and can be replaced by target substructures in the domain augmentation process.

sim_matrix_nb_50.pkl contains a binary matrix M (n_mols x n_mols) that specify the neighbors of each molecule. M[i, j] = 1 indicates j-th molecule is a neighbor of i-th molecule given the neighbor size=50.

The two files can be generated using `pre_structure.py` file.

## Pretrain using domain augmentation and global structure

1. Pretrain using general augmentation, the available augmentations are `['drop_node', 'permute_edge', 'mask_edge', 'subgraph']`. Note that aug1 and aug2 can be different, e.g., `aug1=drop_node, aug2=permute_edge`.

```
python main_cl.py --dataset bace --method local --aug1 drop_node --aug_ratio1 0.2 --aug2 drop_node --aug_ratio2 0.2 --output_model_file pretrain_ --epochs 100 --runseed 0 --lamb 0.0
```

2. Pretrain using domain augmentation, available aug arguments are `[DK1, DK2, DK3, DK5]`.

```
python main_cl.py --dataset bace --method local --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --lamb 0.0
```

3. Pretrain using additional global information by directly supervision

```
python main_cl.py --dataset bace --method global --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --global_mode sup --lamb 1.0
```

4. Pretrain using additional global information by contrastive loss, `n_nb` specifies the neighbor size and the available sizes are `[50, 100, 150, 300]`. 

```
python main_cl.py --dataset bace --method global --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --global_mode cl --n_nb 100 --lamb 1.0
```


## Finetune using pretrained model

1. Linear protocol: only finetune the linear layer on top of GNN using all the labels avaialble. The following commands includes both general augmentations and proposed domain augmentation.

```
python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 1.0 --protocol linear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 1.0 --protocol linear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 1.0 --protocol linear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 1.0 --protocol linear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

```

2. Non-linear (semi-supervised) protocol : finetune all the layers using small fraction of labels. The following commands includes both general augmentations and proposed domain augmentation.

```
python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 0.05 --protocol nonlinear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 100 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 0.05 --protocol nonlinear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 100 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 0.05 --protocol nonlinear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 0.05 --protocol nonlinear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

```

# Saved pretrained models
We also provide pretrained models for each dataset in `/results/<dataset>/` directory. It includes pretrained models from both local contrast and global contrast, which can be used to reproduce the results in the paper. The following table shows the best hyperparameter and performance for the proposed method.

| Dataset      | Augmentation | n_nb | lamb | Linear | Semi-supervised |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| bace		  | DK1         | 50 		  | 5.0 		| 0.845 	  | 0.706		|
| bbbp 		  | DK3         | 50 		  | 10.0 		| 0.905 	  | 0.809		|
| clintox 		  | DK1         | 50 		  | 5.0 		| 0.750 	  | 0.623		|
| mutag 		  | DK3         | 10 		  | 1.0 		| 0.969 	  | 0.916		|
| sider 		  | DK1         | 50 		  | 5.0 		| 0.628 	  | 0.565		|
| tox21 		  | DK1         | 600 		  | 5.0 		| 0.768 	  | 0.686		|
| toxcast 		  | DK1         | 600 		  | 5.0 		| 0.653 	  | 0.546		|

# Acknowledgement

This research is funded by NSF IIS-1749940 (JZ), ONR N00014-20-1-2382 (JZ), NIH 1R01GM134307 (JZ, BC), NIH K01ES028047 (BC).

The backbone of the code is inherited from [Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns).

