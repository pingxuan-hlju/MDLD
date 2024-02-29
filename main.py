import random
import numpy as np
import torch
from models.encoder import PreModel
from utils.load_data import create_feature_matrix, load_data, split_dataset, MyDataset
from utils.evaluate import calculate_AUC_AUPR, evaluate
from torch.utils.data import DataLoader
from utils.params import build_args
import torch.backends.cudnn as cudnn


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    cudnn.enabled = True


def main(args):
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lnc_dis, dis_dis, mi_dis, lnc_mi = load_data()
    # train_index, test_index, val_index = split_dataset(lnc_dis)
    train_index = np.load('G:/Graduate student/Final/Graphormer_DRGCN_01/data/split_dataset/train_index.npy')
    val_index = np.load('G:/Graduate student/Final/Graphormer_DRGCN_01/data/split_dataset/val_index.npy')
    test_index = np.load('G:/Graduate student/Final/Graphormer_DRGCN_01/data/split_dataset/test_index.npy')
    for fold in range(5):
        # fold = 4
        train_set = DataLoader(MyDataset(train_index[fold], lnc_dis), args.batch_size, shuffle=True)
        test_set = DataLoader(MyDataset(test_index, lnc_dis), args.batch_size, shuffle=False)
        val_set = DataLoader(MyDataset(val_index[fold], lnc_dis), args.batch_size, shuffle=False)
        features = create_feature_matrix(lnc_dis, dis_dis, mi_dis, lnc_mi, train_index[fold], device)
        model = PreModel(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_coder, weight_decay=args.lr_wait_coder)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1)
        model.to(device)
        features = features.to(device)
        best = 1e9
        for epoch in range(args.mae_epochs):
            model.train()
            optimizer.zero_grad()
            loss, feat_recon, enc_out, mask_nodes = model(features, epoch=epoch)
            print("--epoch", epoch, "--loss:", loss)
            if loss < best:
                best = loss
                best_model_state_dict = model.state_dict()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        model.load_state_dict(best_model_state_dict)
        model.eval()
        feat = model.get_embeds(features)
        evaluate(feat, train_set, test_set, val_set, args.lr_fc, args.lr_wait_fc, device, fold, train_index,
                 features, args.lr_scheduler, args.early_stopping, args.dropout)


if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)
