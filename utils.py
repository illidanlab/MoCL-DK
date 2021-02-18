import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch_geometric
from torch_geometric.data.dataloader import DataLoader
from torch.autograd import Variable


from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity

# rules
import json
def rule_indicator(smiles_list):
    rules = json.load(open('isostere_transformations_new.json'))
    # rules = json.load(open('rules_carbon_drop.json'))
    print('# rules {:d}'.format(len(rules)))
    rule_indicator = np.zeros([len(smiles_list), len(rules)], dtype=np.int)
    for i in range(len(smiles_list)):
        if i%100==0:
            print(i)
        s = smiles_list[i]
        mol_obj = Chem.MolFromSmiles(s)
        if mol_obj != None:
            for j in range(len(rules)):
                rule = rules[j]
                rxn = AllChem.ReactionFromSmarts(rule['smarts'])
                products = rxn.RunReactants((mol_obj,))
                rule_indicator[i, j] = len(products)
    print(rule_indicator)
    print(np.sum(rule_indicator, axis=1))
    return rule_indicator


def sim_mat(smiles):
    n = len(smiles)
    sim_mat = np.zeros([n, n])
    for i in range(n):
        mol1 = Chem.MolFromSmiles(smiles[i])
        if i%100==0:
            print(i)
        for j in range(n):
            mol2 = Chem.MolFromSmiles(smiles[j])  
            if mol1 != None and mol2 != None:    
                fp1 = AllChem.GetMorganFingerprint(mol1, radius=3)
                fp2 = AllChem.GetMorganFingerprint(mol2, radius=3)
                sim_mat[i, j] = DataStructs.TanimotoSimilarity(fp1, fp2)
    print(sim_mat)
    print('min {:.4f} median {:.4f} max {:.4f}'.format(np.min(sim_mat), np.median(sim_mat), np.max(sim_mat)))
    # sim_mat[sim_mat<=0.4] = 0
    # sim_mat[sim_mat>0.4] = 1
    # print('#1 {:.2f} #total {:d} perc {:.4f}'.format(np.sum(sim_mat), n**2, np.sum(sim_mat)/n**2))
    return sim_mat


def sim_gcn(reps):
    n = reps.shape[0]
    sim_mat = np.zeros([n, n])
    return cosine_similarity(reps, reps)



def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# tag pharmoco features to each atom
fun_smarts = {
        'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
        'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
        'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
        'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
        'Halogen': '[F,Cl,Br,I]'
        }
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def tag_pharmacophore(rdkit_mol_obj):
    for fungrp, qmol in FunQuery.items():
        matches = rdkit_mol_obj.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches: match_idxes.extend(mat)
        for i, atom in enumerate(rdkit_mol_obj.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return rdkit_mol_obj


def atom_feature(atom):
    chirality = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else 'NONE'
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'O', 'N', 'F', 'S', 'Cl', 'P', 'Br', 'I', 'Si', 'Unknown']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3]) +
                    one_of_k_encoding_unk(atom.GetHybridization(),
                                          [Chem.rdchem.HybridizationType.S,
                                           Chem.rdchem.HybridizationType.SP,
                                           Chem.rdchem.HybridizationType.SP2,
                                           Chem.rdchem.HybridizationType.SP3,
                                           Chem.rdchem.HybridizationType.SP3D,
                                           Chem.rdchem.HybridizationType.SP3D2, 'Unknown']) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1, 'Other']) +
                    one_of_k_encoding_unk(chirality, ['R', 'S', 'NONE']) +
                    [int(atom.GetProp('Hbond_donor')),
                     int(atom.GetProp('Hbond_acceptor')),
                     int(atom.GetProp('Basic')),
                     int(atom.GetProp('Acid')),
                     int(atom.GetProp('Halogen')),
                     atom.GetIsAromatic(),
                     atom.IsInRing()]).astype(float).tolist()


# def bond_feature(bond):
#     bt = bond.GetBondType()
#     return np.array([bt == Chem.rdchem.BondType.SINGLE,
#                      bt == Chem.rdchem.BondType.DOUBLE,
#                      bt == Chem.rdchem.BondType.TRIPLE,
#                      bt == Chem.rdchem.BondType.AROMATIC,
#                      bond.GetIsConjugated(),
#                      bond.IsInRing()]).astype(float).tolist()


def bond_feature(bond):
    stereo = bond.GetStereo()
    return np.array([stereo == Chem.rdchem.BondStereo.STEREONONE,
                     stereo == Chem.rdchem.BondStereo.STEREOANY,
                     stereo == Chem.rdchem.BondStereo.STEREOZ,
                     stereo == Chem.rdchem.BondStereo.STEREOE]).astype(float).tolist()


def get_atom_features(mol):
    m = tag_pharmacophore(mol)  # Tag pharmacophore properties of each atom
    atom_list = m.GetAtoms()
    atom_features = []
    for a in atom_list:
        atom_features.append(atom_feature(a))
    atom_features = np.array(atom_features)

    return torch.tensor(atom_features, dtype=torch.float32)


def pad_atom_features(atom_features, max_dim):
    pad_width = max_dim - atom_features.shape[0]
    return np.pad(atom_features, ((0, pad_width), (0, 0)), mode='constant')


def get_bond_features(mol, mono=False):
    m = Chem.MolFromSmiles(mol)
    atom_list = m.GetAtoms()

    bond_features = []
    for i in range(len(atom_list)):
        bond_vector = []
        for j in range(len(atom_list)):
            bond = m.GetBondBetweenAtoms(i, j)
            if mono:
                bf = [float(hasattr(bond, 'GetBondType'))]
            else:
                if hasattr(bond, 'GetBondType'):
                    bf = bond_feature(bond)
                else:
                    bf = [0.0]*4    # change for new bond features
            bond_vector.append(bf)
        bond_features.append(bond_vector)
    return np.array(bond_features)


def add_self_bond(bond_features):
    if len(bond_features.shape) == 3:
        bf = np.transpose(bond_features, (2, 0, 1))
        bf = np.concatenate((bf, [np.identity(bf.shape[2])]), axis=0)
    else:
        bf = np.concatenate(([bond_features], [np.identity(bond_features.shape[1])]), axis=0)
    return bf


def reciprocal_with_zeros(x):
    idx = np.where(x==0.0)
    x_r = np.reciprocal(x, where=(x!=0))
    x_r[idx] = 1.0
    return x_r


def normalize_bond_features(bond_features):
    normalized_bond_features = []
    for adj in bond_features:
        norm_inverse = reciprocal_with_zeros(np.sum(adj, axis=1))
        D = np.diag(norm_inverse)
        normalized_bond_features.append(np.matmul(D, adj))
    return np.array(normalized_bond_features)


def pad_bond_features(bond_features, max_dim):
    pad_width = max_dim - bond_features.shape[1]
    return np.pad(bond_features, ((0, 0), (0, pad_width), (0, pad_width)), mode='constant')


def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
    m = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
    fp_bits = fp.ToBitString()
    finger_print = np.fromstring(fp_bits, 'u1')-ord('0')
    return finger_print


def get_drug_features_batch(drug_smiles, max_dim=200, normalize_edge=False, mono_edge_type=False):
    feature = []
    adj = []
    for mol in drug_smiles:
        af = get_atom_features(mol)
        af = pad_atom_features(af, max_dim)
        feature.append(af)

        bf = get_bond_features(mol, mono=mono_edge_type)
        bf_add_self = add_self_bond(bf)
        if normalize_edge:
            bf_add_self = normalize_bond_features(bf_add_self)
        bf_add_self = pad_bond_features(bf_add_self, max_dim)
        adj.append(bf_add_self)
    feature = np.array(feature)
    adj = np.array(adj)
    return feature, adj


def get_drug_fp_batch(drug_smiles, radius=3, length=1024, FCFP=False):
    fp = []
    for mol in drug_smiles:
        fp.append(get_morgan_fingerprint(mol, radius, length, FCFP))
    fp = np.array(fp)
    return fp


# new
# def get_atom_features(mol):
#     atomic_number = []
#     num_hs = []
    
#     for atom in mol.GetAtoms():
#         atomic_number.append(atom.GetAtomicNum())
#         num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        
#     return torch.tensor([atomic_number, num_hs], dtype=torch.float32).t()


def get_edge_index(mol):
    row, col = [], []
    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        
    return torch.tensor([row, col], dtype=torch.long)


def raw_gene_map():
    fn = './data/output/go_fingerprints_l4.csv'
    res = pd.read_csv(fn)
    return res


from torch.autograd import Variable


def torch_data_drug(mol_list):
    data_list = []

    for i, mol in enumerate(mol_list):

        x1 = get_atom_features(mol)
        edge_index = get_edge_index(mol)

        data = torch_geometric.data.data.Data(x=x1, edge_index=edge_index)
        data_list.append(data)

    return data_list

#

from sklearn.utils import shuffle
import pandas as pd
import random


def get_gene_ft_batch(gene, gene_map):
    gene_name = gene_map['gene']
    #print(gene_name)
    gene_map = gene_map.drop(columns='gene', axis=1)
    gene_map = gene_map.to_numpy()

    #print(gene)
    gene_features = []
    for g in gene:
        idx = np.where(gene_name==g)[0][0]
        gene_features.append(gene_map[idx])

    gene_features = np.array(gene_features)
    #print(gene_features.shape)
    return gene_features.astype(np.float32)


def down_sampling(y):
    print(y.shape)
    unique, counts = np.unique(y, return_counts=True)
    max_idx = np.argmax(counts)
    max_value = unique[max_idx]
    max_counts = counts[max_idx]
    n_select = np.int((np.sum(counts)-max_counts)*0.5)
    print('max_value, max_counts, n_select')
    print(max_value, max_counts, n_select)

    random.seed(0)
    print(np.where(y==max_value)[0])
    idx_select = random.sample(list(np.where(y==max_value)[0]), k=n_select)
    idx_final = np.concatenate([np.where(y==0)[0], idx_select, np.where(y==2)[0]])

    return idx_final


def load_raw(cl='MCF7'):    
    root = './data/output'
    print('process {} trainig data'.format(cl))
    fn = root + '/CL_' + cl + '/data_train.csv'
    table_train = pd.read_csv(fn)
    print(table_train.shape)

    train_labels = table_train['label']
    train_quality = table_train['quality']

    idx_in = down_sampling(train_labels)

    train_smiles = list(table_train['smiles'][idx_in])
    train_genes = list(table_train['gene'][idx_in])
    train_labels = np.asarray(train_labels[idx_in]) # be careful, label index need to be reset using np.array
    train_quality = np.asarray(train_quality[idx_in])
    #train_smiles, train_genes = list(train_smiles), list(train_genes)

    unique, counts = np.unique(train_labels, return_counts=True)
    print(counts)
    print('shuffling training')
    train_smiles, train_genes, train_labels, train_quality = shuffle(train_smiles, 
                                                                    train_genes, 
                                                                    train_labels, 
                                                                    train_quality, 
                                                                    random_state=1)
    print(len(train_smiles), len(train_genes), train_labels.shape, train_quality.shape)
    print('process {} testing data'.format(cl))

    fn = root + '/CL_' + cl + '/data_test.csv'
    table_test = pd.read_csv(fn)
    print(table_test.shape)
    test_smiles = list(table_test['smiles'])
    test_genes = list(table_test['gene'])
    test_labels = np.asarray(table_test['label'])
    test_quality = np.asarray(table_test['quality'])

    #test_smiles, test_genes = list(test_smiles), list(test_genes)

    return train_smiles, train_genes, train_labels, train_quality, test_smiles, test_genes, test_labels, test_quality



if __name__=='__main__':
    d = 'CC#CCOC(=O)C1=CCCN(C1)C'
    mol = Chem.MolFromSmiles(d)
    print(get_atom_features(mol).size())
    print(get_edge_index(mol))

    # smiles_list = ['Cc1cc(c(C)n1c2ccc(F)cc2)S(=O)(=O)NCC(=O)N',
    # 'CN(CC(=O)N)S(=O)(=O)c1c(C)n(c(C)c1S(=O)(=O)N(C)CC(=O)N)c2ccc(F)cc2',
    # 'Fc1ccc(cc1)n2cc(COC(=O)CBr)nn2',
    # 'CCOC(=O)COCc1cn(nn1)c2ccc(F)cc2',
    # 'COC(=O)COCc1cn(nn1)c2ccc(F)cc2',
    # 'Fc1ccc(cc1)n2cc(COCC(=O)OCc3cn(nn3)c4ccc(F)cc4)nn2']


    # cl = 'HT29_t1'
    # load_raw(cl)


