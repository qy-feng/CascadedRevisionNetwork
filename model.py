from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from misc.cap_metrics import create_emb_layer, cal_similarity
from models import att_model


class Captioning(nn.Module):
    def __init__(self, opts):
        super(Captioning, self).__init__()
        self.use_att = opts.use_att
        self.seq_len = opts.seq_len - 1
        self.cap_vocab_size = opts.all_vocab_size
        self.noc_vocab_size = opts.det_class_size
        self.drop_prob = opts.dropout

        # encoder
        self.i2h = nn.Linear(opts.fc_feat_dim, opts.dec_cell_size)
        self.i2c = nn.Linear(opts.fc_feat_dim, opts.dec_cell_size)
        if opts.pre_emb and not opts.evaluate:
            emb_wt_path = os.path.join(opts.data_dir, opts.cap_embeds)
            self.cap_embed = create_emb_layer(emb_wt_path)
        else:
            self.cap_embed = nn.Embedding(self.cap_vocab_size, opts.embedding_size)

        # decoder
        if opts.use_att:
            self.core = att_model.ShowAttendTellCore(opts)
        else:
            self.core = nn.LSTM(opts.embedding_size, opts.dec_cell_size,
                                num_layers=1, batch_first=False)
        self.cap_logit = nn.Linear(opts.dec_cell_size, self.cap_vocab_size)

        # visual match
        self.d2a = nn.Linear(opts.det_feat_dim, opts.dec_cell_size)
        self.h2a = nn.Linear(opts.dec_cell_size, opts.dec_cell_size)
        self.alpha_net = nn.Linear(opts.dec_cell_size, 1)
        self.ppx_logit = nn.Linear(self.noc_vocab_size, 1)

        self.init_weights()

    def init_weights(self):
        init = nn.init.xavier_uniform_
        init(self.i2h.weight)
        init(self.i2c.weight)
        init(self.cap_logit.weight)
        init(self.d2a.weight)
        init(self.ppx_logit.weight)

    def forward(self, feats_fc, feats_att, caps, noc_objs, det_feats, det_values, is_train=True):
        batch_size = feats_fc.size(0)
        feats_fc = feats_fc.squeeze()
        init_hid = self.i2h(feats_fc).unsqueeze(0)
        init_cel = self.i2c(feats_fc).unsqueeze(0)
        state = (init_hid, init_cel)
        det_feats = self.d2a(det_feats)
        det_values = torch.eye(self.noc_vocab_size)[det_values].cuda()

        cap_only = 1

        cap_outs, noc_outs = [], []
        gt_ppx_outs, gf_ppx_outs = [], []
        for i in range(self.seq_len):
            # prepare input
            if i == 0 or is_train is True:
                cap_input = caps[i, :].clone()
            else:
                _, indices = torch.max(cap_out, -1)
                cap_input = indices.long().squeeze()
            xt = self.cap_embed(cap_input)
            xt = F.dropout(xt, self.drop_prob)

            # RNN
            if self.use_att:
                yt, state = self.core(xt, feats_att, state)
            else:
                xt = xt.unsqueeze(0)
                yt, state = self.core(xt, state)

            # caption
            cap_out = self.cap_logit(yt)
            cap_outs.append(cap_out.squeeze(0))

            if not cap_only:
                # visual match
                noc_out, weights = self.vis_match(state[0], det_feats, det_values, batch_size)
                noc_outs.append(noc_out.squeeze())

                # perplexity
                vis_feat = torch.matmul(det_feats.permute(0, 2, 1), weights.permute(0, 2, 1))
                gt_emb = self.cap_embed(caps[i+1, :])
                gt_ppx_out = torch.matmul(gt_emb.unsqueeze(1), vis_feat)
                gt_ppx_out = F.sigmoid(F.softmax(gt_ppx_out, -1))
                gt_ppx_outs.append(gt_ppx_out.squeeze())
                if is_train:
                    gf_emb = self.cap_embed(noc_objs[i+1, :])
                    gf_ppx_out = torch.matmul(gf_emb.unsqueeze(1), vis_feat)
                    gf_ppx_out = F.sigmoid(F.softmax(gf_ppx_out, -1))
                    gf_ppx_outs.append(gf_ppx_out.squeeze())

        cap_outs = torch.stack(cap_outs)
        if not cap_only:
            noc_outs = torch.stack(noc_outs)
            gt_ppx_outs = torch.stack(gt_ppx_outs)
            if is_train:
                gf_ppx_outs = torch.stack(gf_ppx_outs)

        return cap_outs, noc_outs, (gt_ppx_outs, gf_ppx_outs)

        # # visual match
        # att_queries = states
        # if is_train:
        #     att_queries = F.dropout(att_queries, self.drop_prob)
        #     det_feats = F.dropout(det_feats, self.drop_prob)
        #
        # noc_outs = torch.zeros((self.seq_len, batch_size, self.noc_vocab_size)).cuda()
        # att_wgts = torch.zeros((self.seq_len, batch_size, self.noc_vocab_size)).cuda()
        # for i in range(self.seq_len):
        #     att_query = att_queries[i, :, :]
        #     att_predict = self.vis_match(att_query, det_feats, det_values, batch_size)
        #     noc_outs[i, :, :] = att_predict
        #     # att_wgts[i, :, :] = att_weight
        #
        # # perplexity
        # ppx_outs = self.ppx_logit(noc_outs)
        # ppx_outs = F.sigmoid(ppx_outs).squeeze()
        # return cap_outs, ppx_outs, noc_outs

    def vis_match(self, att_query, att_keys, att_values, batch_size):
        # [q] att_query  (batch, dec_cell_size)
        # [k] att_keys (batch, box, dec_cell_size)
        # [v] att_values   (batch, box, class_num)
        att_query = att_query.permute(1, 2, 0)
        weights = torch.matmul(att_keys, att_query)
        weights = F.softmax(weights, -1)
        weights = weights.permute(0, 2, 1)
        att_res = torch.matmul(weights, att_values)
        return att_res, weights

    def attend(self, state, det_feats):
        att_size = det_feats.size(1)
        att_hid_size = state.size(-1)
        att_h = self.h2a(state.squeeze())  # batch * dec_cell_size
        att_h = att_h.unsqueeze(1).expand_as(det_feats)  # batch * att_size * att_hid_size
        dot = det_feats + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot)
        att_res = det_feats.view(-1, att_size, att_hid_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_res).squeeze(1)  # batch * att_feat_size
        return att_res

    # def get_noc_outputs(self, att_states, det_feats, det_values, is_train):
    #     bs, seq_len = att_states.size(1), att_states.size(0)
    #     att_queries = att_states
    #
    #     att_features = self.d2a(det_feats)
    #     att_values = torch.eye(self.noc_vocab_size)[det_values].cuda()
    #     if is_train:
    #         att_queries = F.dropout(att_queries, self.drop_prob)
    #         att_features = F.dropout(att_features, self.drop_prob)
    #
    #     noc_outs = torch.zeros((seq_len, bs, self.noc_vocab_size)).cuda()
    #     att_wgts = torch.zeros((seq_len, bs, self.noc_vocab_size)).cuda()
    #     for i in range(seq_len):
    #         att_query = att_queries[i, :, :]
    #         att_predict, att_weight = self.attention(att_query, att_features, att_values, bs)
    #         noc_outs[i, :, :] = att_predict
    #         att_wgts[i, :, :] = att_weight
    #     return noc_outs, att_wgts  # seq_len, batch, total_noc_words_num
