import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_data():
    lnc_dis = np.loadtxt(r"G:\Graduate student\Final\Graphormer_DRGCN_01\data\LNC\lnc_dis.txt", dtype=int)
    dis_dis = np.loadtxt(r"G:\Graduate student\Final\Graphormer_DRGCN_01\data\LNC\dis_sim.txt", dtype=float)
    mi_dis = np.loadtxt(r"G:\Graduate student\Final\Graphormer_DRGCN_01\data\LNC\mi_dis.txt", dtype=int)
    lnc_mi = np.loadtxt(r"G:\Graduate student\Final\Graphormer_DRGCN_01\data\LNC\lnc_mi.txt", dtype=int)
    return lnc_dis, dis_dis, mi_dis, lnc_mi


def split_dataset(interaction):
    positive_sample = np.argwhere(interaction == 1)
    np.random.shuffle(positive_sample)
    negative_sample = np.argwhere(interaction == 0)
    np.random.shuffle(negative_sample)
    sum_fold = int(positive_sample.shape[0] / 5)
    train_nodes = []
    for i in range(5):
        train_nodes.append(np.vstack((positive_sample[i * sum_fold:(i + 1) * sum_fold],
                                      negative_sample[i * sum_fold:(i + 1) * sum_fold])))
    train_nodes = np.array(train_nodes)
    test_index = []
    for i in range(interaction.shape[0]):
        for j in range(interaction.shape[1]):
            test_index.append([i, j])
    test_index = np.array(test_index)
    train_index = []
    val_index = []
    for i in range(5):
        val_index.append(np.vstack(train_nodes[i]))
        a = [j for j in range(5) if j != i]
        train_index.append(np.vstack((train_nodes[a])))
    train_index = np.array(train_index)
    val_index = np.array(val_index)
    np.save('G:/Graduate student/Final/Graphormer_DRGCN_01/data/split_dataset/train_index.npy', train_index)
    np.save('G:/Graduate student/Final/Graphormer_DRGCN_01/data/split_dataset/val_index.npy', val_index)
    np.save('G:/Graduate student/Final/Graphormer_DRGCN_01/data/split_dataset/test_index.npy', test_index)
    return train_index, test_index, val_index


def split_dataset_final(interaction):
    positive_sample = np.argwhere(interaction == 1)
    np.random.shuffle(positive_sample)
    negative_sample = np.argwhere(interaction == 0)
    np.random.shuffle(negative_sample)
    train_index = np.vstack((positive_sample,
                             negative_sample[0 * positive_sample.shape[0]]))
    train_index = np.array(train_index)
    test_index = []
    for i in range(interaction.shape[0]):
        for j in range(interaction.shape[1]):
            test_index.append([i, j])
    test_index = np.array(test_index)
    return train_index, test_index


class MyDataset(Dataset):
    def __init__(self, matrix, interaction):
        self.matrix = matrix
        self.Interaction = interaction

    def __getitem__(self, idx):
        X, Y = self.matrix[idx]
        label = self.Interaction[X][Y]
        return X, Y, label

    def __len__(self):
        return len(self.matrix)


def create_feature_matrix(lnc_dis, dis_sim, mi_dis, lnc_mi, train_index, device):
    copy_lnc_dis = np.zeros(shape=(lnc_dis.shape[0], lnc_dis.shape[1]), dtype=int)
    for i in range(train_index.shape[0]):
        if lnc_dis[train_index[i][0]][train_index[i][1]] == 1:
            copy_lnc_dis[train_index[i][0]][train_index[i][1]] = 1
    lnc_sim = calculate_sim(lnc_dis, dis_sim)
    mi_sim = calculate_sim(mi_dis, dis_sim)
    # lnc_sim = np.zeros(shape=(240, 240))
    # mi_sim = np.zeros(shape=(495, 495))
    row_1 = np.concatenate((lnc_sim, copy_lnc_dis, lnc_mi), axis=1)
    row_2 = np.concatenate((copy_lnc_dis.T, dis_sim, mi_dis.T), axis=1)
    row_3 = np.concatenate((lnc_mi.T, mi_dis, mi_sim), axis=1)
    features = np.vstack((row_1, row_2, row_3))
    return torch.FloatTensor(features).to(device)


def create_feature_matrix_final(lnc_dis, dis_sim, mi_dis, lnc_mi, device):
    lnc_sim = calculate_sim(lnc_dis, dis_sim)
    mi_sim = calculate_sim(mi_dis, dis_sim)
    # lnc_sim = np.zeros(shape=(240, 240))
    # mi_sim = np.zeros(shape=(495, 495))
    row_1 = np.concatenate((lnc_sim, lnc_dis, lnc_mi), axis=1)
    row_2 = np.concatenate((lnc_dis.T, dis_sim, mi_dis.T), axis=1)
    row_3 = np.concatenate((lnc_mi.T, mi_dis, mi_sim), axis=1)
    features = np.vstack((row_1, row_2, row_3))
    return torch.FloatTensor(features).to(device)


def calculate_sim(interaction, original_sim):
    target_sim = np.zeros(shape=(interaction.shape[0], interaction.shape[0]), dtype=float)
    for i in range(target_sim.shape[0]):
        for j in range(target_sim.shape[1]):
            if i == j:
                target_sim[i][j] = 1
            else:
                l1_num = np.sum(interaction[i] == 1.0)
                l2_num = np.sum(interaction[j] == 1.0)
                if l1_num == 0 or l2_num == 0:
                    target_sim[i][j] = 0
                else:
                    l1_index = np.where(interaction[i] == 1.0)
                    l2_index = np.where(interaction[j] == 1.0)
                    sim_sum = 0.0
                    for l in range(len(l1_index[0])):
                        sim_sum = sim_sum + np.max(original_sim[l1_index[0][l]][l2_index[0]])
                    for l in range(len(l2_index[0])):
                        sim_sum = sim_sum + np.max(original_sim[l2_index[0][l]][l1_index[0]])
                    target_sim[i][j] = sim_sum / (l1_num + l2_num)
    return target_sim


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# lnc_dis, dis_dis, mi_dis, lnc_mi = load_data()
# train_index, test_index = split_dataset_final(lnc_dis)
# features = create_feature_matrix_final(lnc_dis, dis_dis, mi_dis, lnc_mi, device)
# features1 = create_feature_matrix(lnc_dis, dis_dis, mi_dis, lnc_mi, train_index, device)
# print(train_index)
# print(train_index.shape)
# print(test_index)
# print(test_index.shape)
