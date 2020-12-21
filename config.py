import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch Novel Object Captioning')

    # ------------------ data ------------------- #
    parser.add_argument('--data_dir', type=str, default='/data/qfeng/caption/coco',
                        help='root directory of data')
    parser.add_argument('--dataset_path', type=str, default='cap/coco_cap.pkl',
                        help='path of coco dataset')
    parser.add_argument('--info_dic_path', type=str, default='cap/dic_coco.json')
    parser.add_argument('--split_vocab_pkl', type=str, default="noc/coco_split.vocab",
                        help='path of coco dataset')
    parser.add_argument('--cap_embeds', type=str, default='cap/glove/42B/embeds_cap.pt')  # _total
    parser.add_argument('--vis_feat_path', type=str, default='feat/coco_cnn_feature.h5',
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--det_feat_path', default="feat/coco_detection_result.h5",
                        type=str, help='directory of detection feature')
    parser.add_argument('--input_fc_dir', type=str, default='feat/feature_fc',
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='feat/feature_att',
                        help='path to the directory containing the preprocessed att feats')

    # ------------------ model ------------------- #
    parser.add_argument('--fc_feat_dim', default=4096, type=int)
    parser.add_argument('--det_feat_dim', default=1088, type=int)
    parser.add_argument('--all_vocab_size', default=9506, type=int)
    parser.add_argument('--cap_vocab_size', default=9426, type=int)
    parser.add_argument('--det_class_size', default=80, type=int)
    parser.add_argument('--rnn_type', default='LSTM', type=str,
                        help='GRU / LSTM / biLSTM')
    parser.add_argument('--seq_len', default=12, type=int)
    parser.add_argument('--dec_cell_size', default=1024, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--embedding_size', default=1024, type=int)
    parser.add_argument('--att_feat_dim', default=2048, type=int)
    parser.add_argument('--att_hid_size', type=int, default=512,
                        help='the hidden size of the attention MLP')
    parser.add_argument('--max_det_boxes', default=4, type=int, metavar='N')

    # ------------------ strategy ---------------- #
    parser.add_argument('--batch_size', default=32, type=int, metavar='N')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int, metavar='N')
    parser.add_argument('--optim', type=str, default='Adam', help='Adam / SGD')
    parser.add_argument('--momentum', default=0, type=float, metavar='M')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 50, 70, 90])
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.2)

    # ------------------ other ------------------- #
    parser.add_argument('--machine', type=str, default='arc')  # 'uts'
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('-b', '--baseline', dest='baseline', action='store_true',
                        help='train baseline model')
    parser.add_argument('--record', dest='record', action='store_true',
                        help='record model training with tensorboard')
    parser.add_argument('--copy', type=int, default=0)
    parser.add_argument('--subset', type=str, default='mini', help='full / mini / try')
    parser.add_argument('--det_dataset', default='coco', type=str,
                        help='coco / oi / vg')
    parser.add_argument('--init_wt', type=int, default=0)
    parser.add_argument('--sf', type=float, default=0,
                        help='student force probability')
    parser.add_argument('--pre_emb', dest='pre_emb', action='store_true',
                        help='load pretrained embedding weights')
    parser.add_argument('--use_att', dest='use_att', action='store_true',
                        help='train model with attention')

    opts = parser.parse_args()
    return opts
