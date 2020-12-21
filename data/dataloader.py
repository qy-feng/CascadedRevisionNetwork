from __future__ import print_function, division, absolute_import
import torch
from torch.utils import data
import os, sys
import numpy as np
import h5py
import json
import nltk
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from misc.gen_pseudo import get_pseudo_obj

novel_objs = ['bottle', 'bus', 'couch', 'microwave', 'pizza', 'suitcase', 'zebra']  # , 'tennis racket'


class VisualText(data.Dataset):
    def __init__(self, opts, is_train=True, subset='full'):
        # --------------------- parameters -------------------- #
        self.opts = opts
        self.is_train = is_train
        self.seq_len = opts.seq_len
        self.use_att = opts.use_att
        self.cap_vocab_size = opts.cap_vocab_size
        self.det_dataset = opts.det_dataset
        train_data_path = 'cap/cap_info_train.pkl'
        test_data_path = 'cap/cap_info_test.pkl'

        # -------------------- build vocab -------------------- #
        split_vocab_path = os.path.join(opts.data_dir, opts.split_vocab_pkl)
        with open(split_vocab_path, "rb") as f:
            vocab = pkl.load(f)
        self.cap_vocab = vocab["lstm_vocab"]

        if opts.det_dataset == 'coco':
            self.det_vocab = vocab["detcls_vocab"]
        elif opts.det_dataset == 'oi':
            oi_det_feat_dir = os.path.join(opts.data_dir, 'det_feat_oi')
            oi_class_path = os.path.join(oi_det_feat_dir, "open-images-classes.npz")
            self.det_vocab = np.load(oi_class_path)['class_name'].tolist()
            new_num = 0
            for i, word in enumerate(self.det_vocab):
                self.det_vocab[i] = word.lower()
                if word.lower() not in self.cap_vocab:
                    print(word, 'new')
                    new_num += 1
            print(len(self.det_vocab), 'new', new_num)

        self.total_vocab = self.cap_vocab + self.det_vocab

        self.wordtoix = {}
        self.ixtoword = {}
        for idx, word in enumerate(self.total_vocab):
            if word in novel_objs:
                print(idx, word)
            self.wordtoix[word] = idx
            self.ixtoword[idx] = word
        self.BOS_ID = int(self.wordtoix[u'<bos>'])
        self.EOS_ID = int(self.wordtoix[u'<eos>'])
        self.PAD_ID = int(self.wordtoix[u'<pad>'])

        # -------------------- caption set --------------------- #
        if self.is_train:
            with open(os.path.join(opts.data_dir, train_data_path), 'rb') as f:
                self.text_data = pkl.load(f)
            # if subset == 'full':
            #     self.text_data = dataset
            if subset == 'mini':
                self.text_data = self.text_data[:100000]
            elif subset == 'test':
                self.text_data = self.text_data[:1000]
            print("train data number", len(self.text_data))
        else:
            with open(os.path.join(opts.data_dir, test_data_path), 'rb') as f:
                self.text_data = pkl.load(f)
            # if subset == 'full':
            #     self.text_data = dataset
            if subset == 'mini':
                self.text_data = self.text_data[:50000]
            elif subset == 'test':
                self.text_data = self.text_data[:500]
            print("test data number", len(self.text_data))

        # -------------------- image file --------------------- #
        with open(os.path.join(opts.data_dir, opts.info_dic_path), 'r') as f:
            info_dic = json.load(f)
        self.img_ids = {}
        for img in info_dic['images']:
            file_name = img['file_path'].split('/')[1]
            self.img_ids[file_name] = unicode(str(img['id']))

        self.feats_fc_h5 = {}
        # fc_path = os.path.join(opts.data_dir, opts.vis_feat_path)
        # self.input_fc_dir = os.path.join(opts.data_dir, opts.input_fc_dir)
        self.input_fc_dir = '/scratch/qfeng/feature_fc'
        fc_path = os.path.join(self.input_fc_dir, 'feats_fc.h5')
        self.map_shareds(self.text_data, fc_path, self.feats_fc_h5, True)

        if self.use_att:
            self.feats_att_h5 = {}
            # self.input_att_dir = os.path.join(opts.data_dir, opts.input_att_dir)
            self.input_att_dir = '/scratch/qfeng/feature_att'
            att_path = os.path.join(self.input_att_dir, 'feats_att.h5')
            self.map_shareds(self.text_data, att_path, self.feats_att_h5, self.use_att)

        self.feats_det_h5 = {}
        # det_path = os.path.join(opts.data_dir, opts.det_feat_path)
        det_path = '/scratch/qfeng/coco_detection_result.h5'
        self.map_shareds(self.text_data, det_path, self.feats_det_h5)

    def __getitem__(self, index):
        instance = self.text_data[index]
        vname = instance['vname']
        ori_cap = instance['caption_inputs']
        # new_cap = instance['new_captions']
        # novel_obj = instance['novel_objects']
        if len(ori_cap) > self.seq_len - 2:
            cap_target = [self.BOS_ID] + ori_cap[: self.seq_len - 2] + [self.EOS_ID]
        else:
            cap_target = [self.BOS_ID] + ori_cap + [self.EOS_ID]

        noc_target, noc_label = [], []
        noc_object = []
        for i in range(len(cap_target)):
            if cap_target[i] > self.cap_vocab_size:
                noc_object.append(get_pseudo_obj(cap_target[i], self.ixtoword, self.wordtoix))
                noc_target.append(cap_target[i] - self.cap_vocab_size)
                noc_label.append(1)
            else:
                noc_object.append(0)
                noc_target.append(-1)
                noc_label.append(0)

        ix = self.img_ids[vname]
        feat_fc = self.feats_fc_h5[ix][()].astype('float32')
        # feat_fc = self.feats_fc_h5[vname]["fc7"]
        feat_att = None
        if self.use_att:
            feat_att = self.feats_att_h5[ix].value.astype('float32')
        return vname, cap_target, noc_target, noc_label, noc_object, feat_fc, feat_att

    def my_collate_fn(self, data):
        data.sort(key=lambda x: len(x[2]), reverse=True)
        data_values = list(zip(*data))
        vnames = data_values[0]
        batch_size = len(vnames)
        data_keys = ['vnames', 'cap_targets', 'noc_targets', 'noc_labels', 'noc_objs',
                     'cap_lens', 'det_feats', 'det_values', 'feats_fc', 'feats_att']
        batch_data = {'vnames': vnames}
        for key in data_keys[1:5]:
            batch_data[key] = torch.zeros(self.seq_len, batch_size)

        cap_lens = []
        captions = data_values[2]
        for i, cap in enumerate(captions):
            cap_lens.append(len(cap) - 1)
            for k, key in enumerate(data_keys[1:5]):
                batch_data[key][:len(cap), i] = torch.tensor(data_values[k + 1][i])
            # batch_data[data_keys[4]][:len(cap), i, :] = torch.tensor(data_values[k + 1][i, :])

        batch_data['cap_lens'] = cap_lens

        det_feats, det_values = self.get_support_feat(vnames, self.det_dataset)
        batch_data['det_feats'] = torch.tensor(det_feats, dtype=torch.float32)
        batch_data['det_values'] = torch.tensor(det_values).long()
        feats_fc = np.stack(data_values[-2])
        batch_data['feats_fc'] = torch.tensor(feats_fc)
        if self.use_att:
            feats_att = np.stack(data_values[-1])
            batch_data['feats_att'] = torch.tensor(feats_att)

        return batch_data

    def __len__(self):
        return len(self.text_data)

    def get_vocab(self):
        return self.ixtoword

    def get_cap(self):
        # cap = {}
        # for instance in self.text_data:
        #     vname = instance['vname']
        #     if vname not in cap.keys():
        #         cap[vname] = [{'caption': instance['desc']}]
        #     else:
        #         cap[vname].append({'caption': instance['desc']})
        # with open(os.path.join(self.opts.data_dir, 'noc/eval_cap_gt.pkl'), 'wb') as f:
        #     pkl.dump(cap, f)
        with open(os.path.join(self.opts.data_dir, 'cap/eval_cap_gt.pkl'), 'rb') as f:
            cap = pkl.load(f)
        return cap

    def map_shareds(self, meta_info, h5_file, name2shards, use_att=False):
        h = h5py.File(h5_file, 'r')
        d = {}
        for k in h.keys():
            d[k] = True

        for instance in meta_info:
            name = instance['vname']
            if use_att:
                name = self.img_ids[name]
            found = False
            if d.get(name, False):
                name2shards[name] = h[name]
                found = True

            if not found:
                print("key %s not found." % name)
                sys.stdout.flush()
                os._exit(0)

    def get_support_feat(self, vnames, dataset='coco'):
        det_feats = np.zeros([len(vnames), self.opts.max_det_boxes, self.opts.det_feat_dim])
        det_values = np.zeros([len(vnames), self.opts.max_det_boxes])
        for batch_idx, vname in enumerate(vnames):
            if dataset == 'coco':
                feat_file = self.feats_det_h5[vname]
                length = feat_file["value"][:][0].shape[0]
                max_length = min(length, self.opts.max_det_boxes)
                if max_length == 0:  # if no boxes in the image
                    continue
                def_feat = feat_file["feat"][:][0]
                det_value = feat_file["value"][:][0]
            elif dataset == 'oi':
                feat_file = np.load(os.path.join(self.oi_det_feat_dir, vname + '.npz'))
                length = feat_file["det_classes"].shape[0]  # [:][0]
                max_length = min(length, self.opts.max_det_boxes)
                if max_length == 0:  # if no boxes in the image
                    continue
                def_feat = feat_file["det_feature"]
                det_value = feat_file["det_classes"]

            det_feats[batch_idx][:max_length] = def_feat[:max_length, :]
            det_values[batch_idx][:max_length] = det_value[:max_length]
        return det_feats, det_values
