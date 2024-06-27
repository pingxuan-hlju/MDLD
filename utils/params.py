import argparse


def build_args():
    """Set hyperparameters."""

    parser = argparse.ArgumentParser(description="LNC")
    parser.add_argument('--feature_dim', type=int, default=1140, help="number of feature dims")
    parser.add_argument('--model_dim', type=int, default=512, help="number of encoder output dims")
    parser.add_argument('--head_num', type=int, default=8, help="number of heads")
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--layer_num', type=int, default=1, help="number of layers")
    parser.add_argument('--alpha_l', type=int, default=3)
    parser.add_argument('--loss_fn', type=str, default="sce", help="type of loss function")
    parser.add_argument('--dec_in_dim', type=int, default=512, help="number of decoder input dims")
    parser.add_argument('--lr_coder', type=float, default=0.0005, help="learning rate of heterogeneous graph masked"
                                                                       "transformer autoencoder pretraining")
    parser.add_argument('--lr_fc', type=float, default=0.0006, help="learning rate of final model")
    parser.add_argument('--batch_size', type=int, default=32, help="number of batch_size")
    parser.add_argument('--lr_wait_coder', type=float, default=0.0001, help="learning rate decay of heterogeneous "
                                                                            "graph masked transformer autoencoder "
                                                                            "pretraining")
    parser.add_argument('--lr_wait_fc', type=float, default=0.001, help="learning rate decay of final model")
    parser.add_argument('--mae_epochs', type=int, default=400, help="Number of training rounds of hgmt")
    parser.add_argument('--replace_rate', type=float, default=0.0)
    parser.add_argument('--leave_unchanged', type=float, default=0.0)
    parser.add_argument('--feat_mask_rate', type=str, default="0.1,0.005,0.8")
    parser.add_argument('--enc_dec_input_dim', type=int, default=1140, help="number of encoder input dims")
    parser.add_argument('--seed', type=int, default=3407, help="number of seed")
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--early_stopping', type=bool, default=True)
    args = parser.parse_args()
    return args
