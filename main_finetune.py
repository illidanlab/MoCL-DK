import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

from splitters import scaffold_split, random_scaffold_split, random_split
import pandas as pd

import os
import shutil
import datetime
import random

from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train_cls(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()


def train_reg(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_cls(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    print(y_true.shape, y_scores.shape)
    mse = mean_squared_error(y_true, y_scores)
    cor = pearsonr(y_true, y_scores)[0]
    print(mse, cor)
    return mse, cor


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
    parser.add_argument('--dataset', type=str, default = 'esol', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="random", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'dropN_random', help='augmentation1')
    parser.add_argument('--aug2', type=str, default = 'dropN_random', help='augmentation2')
    parser.add_argument('--aug_ratio1', type=float, default = 0.0, help='aug ratio1')
    parser.add_argument('--aug_ratio2', type=float, default = 0.0, help='aug ratio2')
    parser.add_argument('--dataset_load', type=str, default = 'esol', help='load pretrain model from which dataset.')
    parser.add_argument('--protocol', type=str, default = 'linear', help='downstream protocol, linear, nonlinear')
    parser.add_argument('--semi_ratio', type=float, default = 1.0, help='proportion of labels in semi-supervised settings')
    parser.add_argument('--pretrain_method', type=str, default = 'local', help='pretrain_method: local, global')
    parser.add_argument('--lamb', type=float, default = 0.0, help='hyper para of global-structure loss')
    parser.add_argument('--n_nb', type=int, default = 0, help='number of neighbors for  global-structure loss')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag']:
        task_type = 'cls'
    else:
        task_type = 'reg'

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
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'mutag':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print('The whole dataset:', dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # semi-supervised settings
    if args.semi_ratio != 1.0:
        n_total, n_sample = len(train_dataset), int(len(train_dataset)*args.semi_ratio)
        print('sample {:.2f} = {:d} labels for semi-supervised training!'.format(args.semi_ratio, n_sample))
        all_idx = list(range(n_total))
        random.seed(0)
        idx_semi = random.sample(all_idx, n_sample)
        train_dataset = train_dataset[torch.tensor(idx_semi)] #int(len(train_dataset)*args.semi_ratio)
        print('new train dataset size:', len(train_dataset))
    else:
        print('finetune using all data!')


    if args.dataset == 'freesolv':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.pretrain_method == 'local':
        load_dir = 'results/' + args.dataset + '/pretrain_local/'
        save_dir = 'results/' + args.dataset + '/finetune_local/'
    elif args.pretrain_method == 'global':
        load_dir = 'results/' + args.dataset + '/pretrain_global/nb_' + str(args.n_nb) + '/' 
        save_dir = 'results/' + args.dataset + '/finetune_global/nb_' + str(args.n_nb) + '/' 
    else:
        print('Invalid method!!')

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    if not args.input_model_file == "":
        input_model_str = args.dataset_load + '_aug1_' + args.aug1 + '_' + str(args.aug_ratio1) + '_aug2_' + args.aug2 + '_' + str(args.aug_ratio2) + '_lamb_' + str(args.lamb) + '_do_' + str(args.dropout_ratio) + '_seed_' + str(args.runseed)
        output_model_str = args.dataset + '_semi_' + str(args.semi_ratio) + '_protocol_' + args.protocol + '_aug1_' + args.aug1 + '_' + str(args.aug_ratio1) + '_aug2_' + args.aug2 + '_' + str(args.aug_ratio2) + '_lamb_' + str(args.lamb) + '_do_' + str(args.dropout_ratio) + '_seed_' + str(args.runseed) + '_' + str(args.seed)
    else:
        output_model_str = 'scratch_' + args.dataset + '_semi_' + str(args.semi_ratio) + '_protocol_' + args.protocol + '_do_' + str(args.dropout_ratio) + '_seed_' + str(args.runseed) + '_' + str(args.seed)

    txtfile=save_dir + output_model_str + ".txt"
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))  # rename exsist file for collison
    
    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(load_dir + args.input_model_file + input_model_str + '.pth')
        print('successfully load pretrained model!')
    else:
        print('No pretrain! train from scratch!')
    
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

    # if linear protocol, fix GNN layers
    if args.protocol == 'linear':
        print("linear protocol, only train the top layer!")
        for name, param in model.named_parameters():
            if not 'pred_linear' in name:
                param.requires_grad = False
    elif args.protocol == 'nonlinear':
        print("finetune protocol, train all the layers!")
    else:
        print("invalid protocol!")

    # all task info summary
    print('=========task summary=========')
    print('Dataset: ', args.dataset)
    if args.semi_ratio == 1.0:
        print('full-supervised {:.2f}'.format(args.semi_ratio))
    else:
        print('semi-supervised {:.2f}'.format(args.semi_ratio))
    if args.input_model_file == '':
        print('scratch or finetune: scratch')
        print('loaded model from: - ')
    else:
        print('scratch or finetune: finetune')
        print('loaded model from: ', args.dataset_load)
        print('global_mode: n_nb = ', args.n_nb)
    print('Protocol: ', args.protocol)
    print('task type:', task_type)
    print('=========task summary=========')

    # training based on task type
    if task_type == 'cls':
        with open(txtfile, "a") as myfile:
            myfile.write('epoch: train_auc val_auc test_auc\n')
        wait = 0
        best_auc = 0
        patience = 10
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_cls(args, model, device, train_loader, optimizer)

            print("====Evaluation")
            if args.eval_train:
                train_auc = eval_cls(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_auc = 0
            val_auc = eval_cls(args, model, device, val_loader)
            test_auc = eval_cls(args, model, device, test_loader)

            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_auc) + ' ' + str(val_auc) + ' ' + str(test_auc) + "\n")

            print("train: %f val: %f test: %f" %(train_auc, val_auc, test_auc))

            # Early stopping
            if np.greater(val_auc, best_auc):  # change for train loss
                best_auc = val_auc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print('Early stop at Epoch: {:d} with final val auc: {:.4f}'.format(epoch, val_auc))
                    break

    elif task_type == 'reg':
        with open(txtfile, "a") as myfile:
            myfile.write('epoch: train_mse train_cor val_mse val_cor test_mse test_cor\n')
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train(args, model, device, train_loader, optimizer)

            print("====Evaluation")
            if args.eval_train:
                train_mse, train_cor = eval_reg(args, model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_mse, train_cor = 0, 0
            val_mse, val_cor = eval_reg(args, model, device, val_loader)
            test_mse, test_cor = eval_reg(args, model, device, test_loader)

            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_mse) + ' ' + str(train_cor) + ' ' + str(val_mse) + ' ' + str(val_cor) + ' ' + str(test_mse) + ' ' + str(test_cor) + "\n")

            print("train: %f val: %f test: %f" %(train_mse, val_mse, test_mse))
            print("train: %f val: %f test: %f" %(train_cor, val_cor, test_cor))


if __name__ == "__main__":
    main()
