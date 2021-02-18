# MoCL: Contrastive Learning on Molecular Graphs withMulti-level Domain Knowledge
## Abstract
Recent years have seen a rapid growth of utilizing graph neural net-works (GNNs) in the biomedical domain for tackling drug-relatedproblems. However, like any other deep architectures, GNNs aredata hungry. While requiring labels in real world is often expen-sive, pretraining GNNs in an unsupervised manner has been ac-tively explored. Among them, graph contrastive learning, by max-imizing the mutual information between paired graph augmenta-tions, has been shown to be effective on various downstream tasks.However, the current graph contrastive learning framework hastwo limitations. First, the augmentations are designed for generalgraphs and thus may not be suitable or powerful enough for cer-tain domains. Second, the contrastive scheme only learns repre-sentations that are invariant to local perturbations and thus doesnot consider the global structure of the dataset, which may alsobe useful for downstream tasks. Therefore, in this paper, we studygraph contrastive learning in the context of biomedical domain,where molecular graphs are present. We propose a novel frame-work called MoCL, which utilizes domain knowledge at both local-and global-level to assist representation learning. The local-leveldomain knowledge guides the augmentation process such that vari-ation is introduced without changing graph semantics. The global-level knowledge encodes the similarity information between graphsin the entire dataset and helps to learn representations with richersemantics. The entire model is learned through a double contrastobjective. We evaluate MoCL on various molecular datasets un-der both linear and semi-supervised settings and results show thatMoCL achieves state-of-the-art performance.

The current version of the paper can be found here (TBD).

## Compatibility
The code is compatible with python 3.7

## Pretrain using domain augmentation and global structure

python main_cl.py --dataset bace --method local --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --lamb 0.0 

python main_cl.py --dataset bace --method global --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --global_mode sup --lamb 1.0 

python main_cl.py --dataset bace --method global --aug1 DK1 --aug2 DK1 --output_model_file pretrain_ --epochs 100 --runseed 0 --global_mode cl --n_nb 100 --lamb 1.0 


## finetune using pretrained model

