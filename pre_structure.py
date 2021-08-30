import argparse
import numpy as np
import pandas as pd
import os
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
import pickle as pkl


def stats(smiles_list):
    # statics of the dataset
    n_atoms = [0] * len(smiles_list)
    avg_d = [0] * len(smiles_list)
    for i in range(len(smiles_list)):
        s = smiles_list[i]
        mol = Chem.MolFromSmiles(s)
        if mol != None:
            n = len(mol.GetAtoms())
            e = len(mol.GetBonds())
            n_atoms[i]=n
            avg_d[i] = e

    #print(n_atoms)
    print(np.min(n_atoms), np.median(n_atoms), np.mean(n_atoms), np.max(n_atoms))
    print(np.min(avg_d), np.median(avg_d), np.mean(avg_d), np.max(avg_d))
    with open(save_dir + 'stats.pkl', 'wb') as f:
        pkl.dump([n_atoms, avg_d], f)


import heapq
def nearest_neighbor(sim_mat, n_nb):
    n = sim_mat.shape[0]
    res = np.zeros([n, n])
    for i in range(n):
        r = sim_mat[i, :]
        idx = heapq.nlargest(n_nb, range(len(r)), r.take)
        res[i, idx] = 1
        if i%100==0:
            print(i)
    res_sum_v = np.sum(res, axis=1)
    print('row sum: ', res_sum_v)
    d = res.diagonal()
    print('diagonal: ', d)
    return res

# other functions
def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
    m = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
    fp_bits = fp.ToBitString()
    finger_print = np.fromstring(fp_bits, 'u1')-ord('0')
    return finger_print

def get_drug_fp_batch(drug_smiles, radius=3, length=1024, FCFP=False):
    fp = []
    for mol in drug_smiles:
        fp.append(get_morgan_fingerprint(mol, radius, length, FCFP))
    fp = np.array(fp)
    return fp


def main():
    parser = argparse.ArgumentParser(description='Get domain rule indicator and neighbor matrix')
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    args = parser.parse_args()

    if args.dataset == "tox21": #8k
        num_tasks = 12
    elif args.dataset == "hiv": #40k
        num_tasks = 1
    elif args.dataset == "pcba": #400k not used
        num_tasks = 128
    elif args.dataset == "muv": #90k
        num_tasks = 17
    elif args.dataset == "bace": #1.5k
        num_tasks = 1
    elif args.dataset == "bbbp": #2k
        num_tasks = 1
    elif args.dataset == "toxcast": #8k
        num_tasks = 617
    elif args.dataset == "sider":   #1427
        num_tasks = 27
    elif args.dataset == "clintox": #1491
        num_tasks = 2
    elif args.dataset == 'esol':    #1128
        num_tasks = 1
    elif args.dataset == 'mutag':
        num_tasks = 1
    elif args.dataset == 'dti':
        num_tasks = 0
    elif args.dataset == 'moonshot':    #other projects
        num_tasks = 1
    elif args.dataset == 'ncats':   #other projects
        num_tasks = 1
    elif args.dataset == 'mooncats':    #other projects
        num_tasks = 1
    elif 'linc' in args.dataset:
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    #dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    if args.dataset == 'bbbp':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset.upper() +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'bace':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
        smiles_list = input_df['mol'].tolist()
    elif args.dataset == 'clintox':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'sider':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'tox21':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'toxcast':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'_data.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'muv':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'hiv':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset.upper() +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'mutag':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset + '_188_data.can', sep=' ', header=None)
        smiles_list = input_df[0].tolist()
    elif args.dataset == 'dti':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/' + args.dataset +'.csv', sep=',')
        smiles_list = input_df['smiles'].tolist()
    elif args.dataset == 'moonshot':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/3CL_Moonshot_activity_data_prep.csv', sep=',')
        smiles_list = input_df['SMILES'].tolist()
    elif args.dataset == 'ncats':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/3CL_NCATS_enzymatic_activity_prep.csv', sep=',')
        smiles_list = input_df['SMILES'].tolist()
    elif args.dataset == 'mooncats':
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/mooncats_smiles.csv', sep=',')
        smiles_list = input_df['SMILES'].tolist()
    elif 'lincs' in args.dataset:
        input_df = pd.read_csv("dataset/" + args.dataset + '/raw/unique_drugs.csv', sep=',')
        smiles_list = input_df['SMILES'].tolist()
    else:
        print('original smiles list not found!')

    print('smiles length {:d}'.format(len(smiles_list)))
    save_dir = 'results/' + args.dataset + '/'

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    # rule indicator
    from utils import rule_indicator, sim_mat
    rule_indicator = rule_indicator(smiles_list)
    with open(save_dir + 'rule_indicator_new.pkl', 'wb') as f:
        pkl.dump([rule_indicator], f)

    # similarity matrix
    sim_matrix = sim_mat(smiles_list)
    with open(save_dir + 'sim_matrix.pkl', 'wb') as f:
        pkl.dump([sim_matrix], f)

    # neighbor matrix based on sim_matrix and n_nb (nearest neighbor size)
    if args.dataset in ['tox21', 'toxcast']:
        n_nb_list = [600, 800, 1000]
    else:
        n_nb_list = [10, 50, 100, 150, 300]

    with open(save_dir + 'sim_matrix.pkl', 'rb') as f:
        df = pkl.load(f)
        sim_matrix = df[0]
    print(np.min(sim_matrix), np.max(sim_matrix))
    print(np.sum(sim_matrix, axis=1))
    for n_nb in n_nb_list:
        print('generate nb: ', n_nb)
        sim_matrix_idx = nearest_neighbor(sim_matrix, n_nb)
        with open(save_dir + 'sim_matrix_nb_' + str(n_nb) + '.pkl', 'wb') as f:
            pkl.dump([sim_matrix_idx], f)


if __name__ == "__main__":
    main()
