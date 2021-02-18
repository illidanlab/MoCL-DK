import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred, GNN_graphCL
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_scaffold_split, random_split
import pandas as pd

import os
import shutil
import datetime

from tensorboardX import SummaryWriter

from copy import deepcopy
import pickle as pkl
import json

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train_base(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2):

    dataset.aug = "none"
    dataset = dataset.shuffle()
    dataset1 = deepcopy(dataset)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2 in zip(loader1, loader2):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward(data1)
        out2 = model.forward(data2)
        loss = model.loss_cl(out1, out2)
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset), 0


def train_global(model, optimizer, dataset, device, batch_size, 
    aug1, aug_ratio1, aug2, aug_ratio2, sim_global, sim_global_nb, lamb, mode='sup'):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)

    dataset3 = deepcopy(dataset1)
    dataset3.aug = 'none'
    loader3 = DataLoader(dataset3, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2, data3 in zip(loader1, loader2, loader3):
        #print(data1, data2, data3)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward(data1)
        out2 = model.forward(data2)

        loss = model.loss_cl(out1, out2)

        data3 = data3.to(device)
        out3 = model.forward(data3)
        #print('data3 idx',data3.id)
        if mode=='sup':
            loss2 = model.loss_global_sup(out3, out3, data3.id, sim_global)
        elif mode=='cl':
            loss2 = model.loss_global_cl(out3, data3.id, sim_global_nb)
        else:
            print('invalid mode!')
        #print('loss1 {:.4f} loss2 {:.4f}'.format(loss, loss2))
        loss += lamb * loss2

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset), 0



from loader import mol_to_graph_data_obj_simple
from torch_geometric.data import Batch
import random
from rdkit import Chem
from rdkit.Chem import AllChem
def generate_aug(data_index, smiles_list, rule_indicator, rules):
    data_list = []
    for row_idx in data_index:
        #print(row_idx)
        s = smiles_list[row_idx]
        mol_obj = Chem.MolFromSmiles(s)
        non_zero_idx = np.where(rule_indicator[row_idx, :]!=0)[0]
        #print('non_zero_idx len:', len(non_zero_idx))
        if len(non_zero_idx)==0:
            mol = mol_obj
        else:
            # random pick one rule
            col_idx = random.choice(non_zero_idx)
            # pick one aug
            #print('row idx {:d} col_idx {:}'.format(row_idx, col_idx))
            #print('# rules ', rule_indicator[row_idx, col_idx])
            aug_idx = random.choice(range(rule_indicator[row_idx, col_idx]))
            #print('aug_idx: ', aug_idx)

            rule = rules[col_idx]
            rxn = AllChem.ReactionFromSmarts(rule['smarts'])
            products = rxn.RunReactants((mol_obj,))
            #print('products', products)
            #print('products len', len(products))
            mol = products[aug_idx][0]
            #Chem.SanitizeMol(mol)
        data = mol_to_graph_data_obj_simple(mol)
        data_list.append(data)
    #print(data_list)
    batch = Batch.from_data_list(data_list)
    #print(batch)
    return batch


def train_intra(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2, 
    smiles_list, rule_indicator, rules):

    dataset.aug = "none"
    dataset = dataset.shuffle()
    loader = DataLoader(dataset, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        #print(data)
        optimizer.zero_grad()
        data_index = data.id
        data1 = generate_aug(data_index, smiles_list, rule_indicator, rules)
        data2 = generate_aug(data_index, smiles_list, rule_indicator, rules)

        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward(data1)
        out2 = model.forward(data2)
        loss = model.loss_cl(out1, out2)

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader.dataset), 0


def train_full(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2, 
    smiles_list, rule_indicator, rules, sim_global, lamb):

    dataset.aug = "none"
    dataset = dataset.shuffle()
    loader = DataLoader(dataset, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        #print(data)
        optimizer.zero_grad()
        data_index = data.id
        data1 = generate_aug(data_index, smiles_list, rule_indicator, rules)
        data2 = generate_aug(data_index, smiles_list, rule_indicator, rules)

        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward(data1)
        out2 = model.forward(data2)
        loss = model.loss_cl(out1, out2)

        data = data.to(device)
        out = model.forward(data)
        loss2 = model.loss_cl_inter(out, out, data.id, sim_global)
        #print('loss1 {:.4f} loss2 {:.4f}'.format(loss, loss2))
        loss += lamb * loss2

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader.dataset), 0



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'drop_node', help='augmentation1')
    parser.add_argument('--aug2', type=str, default = 'drop_node', help='augmentation2')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2, help='aug ratio1')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2, help='aug ratio2')
    parser.add_argument('--method', type=str, default = 'local', help='method: local, global')
    parser.add_argument('--lamb', type=float, default = 0.0, help='hyper para of global-structure loss')
    parser.add_argument('--n_nb', type=int, default = 0, help='number of neighbors for  global-structure loss')
    parser.add_argument('--global_mode', type=str, default = 'sup', help='global mode: sup or cl')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'mutag':
        num_tasks = 1
    elif args.dataset == 'dti':
        num_tasks = 0
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    # if args.dataset in ['bace']:
    #     smiles_list = dataset.smiles[0].tolist()    # smiles list should be smiles before processing, not the dataset smiels.
    #     assert len(dataset) == len(smiles_list)
    # elif args.dataset in ['bbbp', 'clintox', 'sider']:
    #     if args.dataset == 'bbbp':
    #         raw_file = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset.upper() +'.csv', sep=',')
    #     else: 
    #         raw_file = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
    #     smiles_list = raw_file['smiles'].tolist()
    # print('dataset smiles {:d} raw smiles {:d}'.format(len(dataset), len(smiles_list)))
    print(dataset)
    print(dataset.data)
    

    #print(smiles_list)
    # sim_mat = np.zeros([len(dataset), len(dataset)])
    # sim_mat = torch.from_numpy(sim_mat).to(device)
    # smiles = dataset.smiles.iloc[:, 0].tolist()
    # A = sim_mat(smiles)
    # D = np.diag(A.sum(axis=1))
    # L = D-A
    # vals, vecs = np.linalg.eig(L)
    # vecs = vecs[:,np.argsort(vals)]
    # vals = vals[np.argsort(vals)]
    # print(vals)
    # print(np.min(vals), np.median(vals), np.max(vals))

    if args.method == 'local':
        save_dir = 'results/' + args.dataset + '/pretrain_local/'
    elif args.method == 'global':
        save_dir = 'results/' + args.dataset + '/pretrain_global/nb_' + str(args.n_nb) + '/' 
        if args.dataset == 'hiv':
            sim_matrix = np.zeros([len(dataset.original_smiles), len(dataset.original_smiles)])
            sim_matrix_nb = np.zeros([len(dataset.original_smiles), len(dataset.original_smiles)])
        else:
            if args.global_mode == 'cl':
                with open('results/'+args.dataset+'/sim_matrix_nb_'+str(args.n_nb)+'.pkl', 'rb') as f:
                    df = pkl.load(f)
                    sim_matrix_nb = df[0]
                sim_matrix_nb = torch.from_numpy(sim_matrix_nb).to(device)
                print('sim_matrix_nb loaded with size: ', sim_matrix_nb.size())
                sim_matrix = None
            elif args.global_mode == 'sup':
                with open('results/'+args.dataset+'/sim_matrix.pkl', 'rb') as f:
                    df = pkl.load(f)
                    sim_matrix = df[0]
                sim_matrix = torch.from_numpy(sim_matrix).to(device)
                print('sim_matrix loaded with size: ', sim_matrix.size())
                sim_matrix_nb = None

    else:
        print('Invalid method!!')

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    model_str = args.dataset + '_aug1_' + args.aug1 + '_' + str(args.aug_ratio1) + '_aug2_' + args.aug2 + '_' + str(args.aug_ratio2) + '_lamb_' + str(args.lamb) + '_do_' + str(args.dropout_ratio) + '_seed_' + str(args.runseed)

    txtfile=save_dir + model_str + ".txt"
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))  # rename exsist file for collison
    
    #set up model
    model = GNN_graphCL(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_loss\n')

    rules = json.load(open('isostere_transformations_new.json'))
    with open('results/'+ args.dataset + '/rule_indicator_new.pkl', 'rb') as f:
        d = pkl.load(f)
        rule_indicator = d[0]
    print('rule indicator shape: ', rule_indicator.shape)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        if args.method == 'local':
            train_loss, _ = train_base(model, optimizer, dataset, device, args.batch_size, args.aug1, args.aug_ratio1, args.aug2, args.aug_ratio2)
        elif args.method == 'global':
            train_loss, _ = train_global(model, optimizer, dataset, device, args.batch_size, args.aug1, args.aug_ratio1, args.aug2, args.aug_ratio2, 
                sim_matrix, sim_matrix_nb, args.lamb, mode=args.global_mode)
        else:
            print('invalid method!!')
        
        print("train: %f" %(train_loss))

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': ' + str(train_loss) + "\n")

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), save_dir + args.output_model_file + model_str + ".pth")


if __name__ == "__main__":
    main()
