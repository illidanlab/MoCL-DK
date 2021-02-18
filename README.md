# MoCL: Contrastive Learning on Molecular Graphs with Multi-level Domain Knowledge
## Abstract
In this paper, we study graph contrastive learning in the context of biomedical domain,where molecular graphs are present. We propose a novel frame-work called MoCL, which utilizes domain knowledge at both local-and global-level to assist representation learning. The local-leveldomain knowledge guides the augmentation process such that vari-ation is introduced without changing graph semantics. The global-level knowledge encodes the similarity information between graphsin the entire dataset and helps to learn representations with richersemantics. The entire model is learned through a double contrastobjective. We evaluate MoCL on various molecular datasets un-der both linear and semi-supervised settings and results show thatMoCL achieves state-of-the-art performance.

The current version of the paper can be found here (TBD).

## Installation
The code are tested on python 3.7 and following are the dependencies:

```
pytorch = 1.6.0
cuda = 10.2
torch-geometric (need to specify torch and cuda version during installation)
rdkit (most recent)
tqdm (most recent)
tensorboardx (most recent)
```

## Pretrain using domain augmentation and global structure

Pretrain using general augmentation, the available augmentations are `['drop_node', 'permute_edge', 'mask_edge', 'subgraph']`. Note that aug1 and aug2 can be different, e.g., `aug1=drop_node, aug2=permute_edge`.

```
python main_cl.py --dataset bace --method local --aug1 drop_node --aug_ratio1 0.2 --aug2 drop_node --aug_ratio2 0.2 --output_model_file pretrain_ --epochs 100 --runseed 0 --lamb 0.0
```

Pretrain using domain augmentation, available aug arguments are `[DK1, DK2, DK3, DK5]`.

```
python main_cl.py --dataset bace --method local --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --lamb 0.0
```

Pretrain using additional global information by directly supervision

```
python main_cl.py --dataset bace --method global --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --global_mode sup --lamb 1.0
```

Pretrain using additional global information by contrastive loss, `n_nb` specifies the neighbor size and the available sizes are `[50, 100, 150, 300]`. 

```
python main_cl.py --dataset bace --method global --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --global_mode cl --n_nb 100 --lamb 1.0
```


## Finetune using pretrained model

linear protocol: only finetune the linear layer on top of GNN using all the labels avaialble. The following commands includes both general augmentations and proposed domain augmentation.

```
python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 1.0 --protocol linear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 1.0 --protocol linear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 1.0 --protocol linear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 1.0 --protocol linear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

```

non-linear (semi-supervised) protocol : finetune all the layers using small fraction of labels. The following commands includes both general augmentations and proposed domain augmentation.

```
python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 0.05 --protocol nonlinear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 100 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method local --semi_ratio 0.05 --protocol nonlinear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 100 --runseed 0 --seed 0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 0.05 --protocol nonlinear --aug1 drop_node --aug_ratio1 0.20 --aug2 drop_node --aug_ratio2 0.20 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

python main_finetune.py --dataset bace --dataset_load bace --pretrain_method global --n_nb 100 --semi_ratio 0.05 --protocol nonlinear --aug1 DK1 --aug2 DK1 --input_model_file pretrain_ --epochs 50 --runseed 0 --seed 0 --lamb 1.0

```

# Acknowledgement

The backbone of the code is inherited from [Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns).

