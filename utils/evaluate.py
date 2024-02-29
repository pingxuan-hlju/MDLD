import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_auc_score, average_precision_score
from Graphormer_DRGCN_01.models.test import Test
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from Graphormer_DRGCN_01.utils.lr import LRScheduler
from Graphormer_DRGCN_01.utils.early_stop import EarlyStopping


def fold_5(TPR, FPR, PR):
    fold = len(TPR)
    le = []
    for i in range(fold):
        le.append(len(TPR[i]))
    min_f = min(le)
    F_TPR = np.zeros((fold, min_f))
    F_FPR = np.zeros((fold, min_f))
    F_P = np.zeros((fold, min_f))
    for i in range(fold):
        k = len(TPR[i]) / min_f
        for j in range(min_f):
            F_TPR[i][j] = TPR[i][int(round(((j + 1) * k))) - 1]
            F_FPR[i][j] = FPR[i][int(round(((j + 1) * k))) - 1]
            F_P[i][j] = PR[i][int(round(((j + 1) * k))) - 1]
    TPR_5 = F_TPR.sum(0) / fold
    FPR_5 = F_FPR.sum(0) / fold
    PR_5 = F_P.sum(0) / fold
    return TPR_5, FPR_5, PR_5


def calculate_TPR_FPR(RD, f, B):
    old_id = np.argsort(-RD)
    min_f = int(min(f))
    max_f = int(max(f))
    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)
    for i in range(RD.shape[0]):
        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
    ki = 0
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]
            FP[i] = FP[i] / FP_TN[i]
    for i in range(RD.shape[0]):
        kk = f[i] / min_f
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(np.round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(np.round_(((j + 1) * kk))) - 1]
    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    P = P2.sum(0) / (P.shape[0] - ki)
    return TPR, FPR, P


def curve(FPR, TPR, P, TPRs, FPRs, Ps):
    plt.figure()
    plt.subplot(121)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    n = len(TPRs)
    # plt.title("ROC curve  (AUC = %.4f)" % (auc(FPR, TPR)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    for i in range(0, n):
        plt.plot(FPRs[i], TPRs[i], label="fold=%d  AUC=%.3f" % (i, auc(FPRs[i], TPRs[i])))
        plt.legend(loc='lower right')
    plt.plot(FPR, TPR, label="mean AUC=%.3f" % (auc(FPR, TPR)))
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    # plt.title("PR curve  (AUPR = %.4f)" % (auc(TPR, P) + (TPR[0] * P[0])))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    for i in range(0, n):
        plt.plot(TPRs[i], Ps[i], label="fold=%d  AUPR=%.3f" % (i, (auc(TPRs[i], Ps[i]) + (TPRs[i][0] * Ps[i][0]))))
        plt.legend(loc='lower right')
    plt.plot(TPR, P, label="mean AUPR=%.3f" % (auc(TPR, P) + (TPR[0] * P[0])))
    plt.legend(loc='lower right')
    plt.show()


def calculate_AUC_AUPR():
    FPRs = []
    TPRs = []
    Ps = []
    for step in range(5):
        # step = 0
        R = np.load(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\score\score_%d.npy' % step)
        label = np.loadtxt(r"G:\Graduate student\Final\Graphormer_DRGCN_01\data\LNC\lnc_dis.txt", dtype=int)
        index = np.load(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\index\index_%d.npy' % step)
        # auc = roc_auc_score(label, R)
        # aupr = average_precision_score(label, R)
        # print(auc)
        # print(aupr)
        for i in range(index.shape[0]):
            R[index[i][0]][index[i][1]] = -1
            label[index[i][0]][index[i][1]] = -1
        # R = np.transpose(R)
        # label = np.transpose(label)
        f = np.zeros(shape=(R.shape[0], 1))
        for i in range(R.shape[0]):
            f[i] = np.sum(R[i] > -1)
        TPR, FPR, P = calculate_TPR_FPR(R, f, label)
        # print(TPR.shape, FPR.shape, P.shape)
        FPRs.append(FPR)
        TPRs.append(TPR)
        Ps.append(P)
    TPR_5, FPR_5, PR_5 = fold_5(TPRs, FPRs, Ps)
    np.savetxt(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\TPR.txt', TPR_5)
    np.savetxt(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\FPR.txt', FPR_5)
    np.savetxt(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\P.txt', PR_5)
    curve(FPR_5, TPR_5, PR_5, TPRs, FPRs, Ps)


def evaluate(feat, train_set, test_set, val_set, lr, wd, device, fold, train_index, features,
             lr_scheduler, early_stopping, dropout):
    net = Test(dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss_function = nn.CrossEntropyLoss()
    if lr_scheduler:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
    if early_stopping:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
    for j in range(300):
        net.train()
        for step, (x, y, label) in enumerate(train_set):
            label = Variable(label).long().to(device)
            optimizer.zero_grad()
            # pair_emb = torch.cat((feat[x], feat[y + 240]), 1).to(device)
            pre = net(feat, features, x, y)
            train_loss = loss_function(pre, label)
            train_loss.backward()
            optimizer.step()
            print('-fold:', fold, '-epoch:', j, '-batch:', step, '-loss:', train_loss)
        net.eval()
        loss = 0.000000000000000
        for _, (x, y, label) in enumerate(val_set):
            x = x.type(torch.long)
            y = y.type(torch.long)
            label = Variable(label).long().to(device)
            with torch.no_grad():
                pre = net(feat, features, x, y)
                val_loss = loss_function(pre, label)
                print("--val_loss", val_loss)
                loss += val_loss.item()
        # if lr_scheduler:
        #     lr_scheduler(loss)
        if early_stopping:
            early_stopping(loss)
            if early_stopping.early_stop:
                break
    net.eval()  # 测试
    score = np.full((240, 405), 0, dtype=float)
    for _, (x, y, _) in enumerate(test_set):
        x = x.type(torch.long)
        y = y.type(torch.long)
        with torch.no_grad():
            pre = net(feat, features, x, y)
        pre = F.softmax(pre, dim=1)
        for index in range(pre.shape[0]):
            score[x[index]][y[index]] = pre[index][1]
        # 保存训练集索引,模型和得分矩阵
    np.save(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\index\index_%d.npy' % fold, train_index[fold])
    np.save(r'G:\Graduate student\Final\Graphormer_DRGCN_01\utils\result\score\score_%d.npy' % fold, score)


# calculate_AUC_AUPR()
