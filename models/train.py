import numpy as np
import torch
from .model import Model
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from utils.lr import LRScheduler
from utils.early_stop import EarlyStopping


def train(feat, train_set, test_set, val_set, lr, wd, device, fold, train_index, features,
          lr_scheduler, early_stopping, dropout):
    """Train and test"""

    net = Model(dropout).to(device)
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
        if lr_scheduler:
            lr_scheduler(loss)
        if early_stopping:
            early_stopping(loss)
            if early_stopping.early_stop:
                break
    # testing
    net.eval()
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
    np.save(r'G:\Graduate student\Final\MDLD\utils\result\index\index_%d.npy' % fold, train_index[fold])
    np.save(r'G:\Graduate student\Final\MDLD\utils\result\score\score_%d.npy' % fold, score)
