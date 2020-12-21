from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import os
import time
from numpy import random
from misc import metric
from misc import utils


def train(train_loader, model, optimizer, opts, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.train()
    end = time.time()
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        feats_fc = batch_data['feats_fc'].cuda()
        det_feats = batch_data['det_feats'].cuda()
        det_values = batch_data['det_values'].cuda()
        lengths = batch_data['cap_lens']
        tgt_caps = batch_data['cap_targets'].long().cuda()
        noc_labels = batch_data['noc_labels'].cuda()
        noc_targets = batch_data['noc_targets'].long().cuda()
        noc_objs = batch_data['noc_objs'].long().cuda()
        # cap_inputs = tgt_caps[:-1, :]
        # cap_targets = tgt_caps[1:, :]

        if not opts.use_att:
            cap_outs, noc_outs, ppx_outs = model(feats_fc, None, tgt_caps, noc_objs,
                                                 det_feats, det_values)
        else:
            feats_att = batch_data['feats_att'].cuda()
            cap_outs, noc_outs, ppx_outs = model(feats_fc, feats_att, tgt_caps, noc_objs,
                                                 det_feats, det_values)

        cap_loss = metric.pack_crit(cap_outs, tgt_caps[1:, :], lengths)

        cap_only = 1
        if cap_only:
            loss = cap_loss
        else:
            noc_loss = metric.mask_crit(noc_outs, noc_targets[1:, :], noc_labels[1:, :])
            ppx_loss = metric.mask_crit(ppx_outs[0].unsqueeze(2), noc_labels[1:, :]*0, noc_labels[1:, :], crit='B') + \
                       metric.mask_crit(ppx_outs[1].unsqueeze(2), noc_labels[1:, :], noc_labels[1:, :], crit='B')
            loss = (cap_loss + noc_loss + ppx_loss) / 4

        losses.update(loss.item(), feats_fc.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % (len(train_loader)//5 + 1) == 0:
            if opts.baseline:
                print("Epoch {}({}), train_loss = {:.3f}"
                      " Time {batch_time.val:.3f} Data {data_time.val:.3f}" \
                      .format(epoch, i, losses.avg,
                              batch_time=batch_time, data_time=data_time))
            else:
                if cap_only:
                    print("Epoch {}({}), train_loss = {:.3f}" \
                          .format(epoch, i, losses.avg))
                else:
                    print("Epoch {}({}), train_loss = {:.3f} cap {:.3f} cls {:.3f} noc {:.3f},"
                          " Time {batch_time.val:.3f} Data {data_time.val:.3f}" \
                          .format(epoch, i, losses.avg, cap_loss.item(), ppx_loss.item(), noc_loss.item(),
                                  batch_time=batch_time, data_time=data_time))

    return losses.avg


def evaluate(eval_loader, model, opts, vocab, cap_gt, epoch):
    model.eval()

    results_wo_novel, names_wo_novel = [], []
    results_total, names_total = [], []

    for i, batch_data in enumerate(eval_loader):
        vnames = batch_data['vnames']
        det_feats = batch_data['det_feats'].cuda()
        det_values = batch_data['det_values'].cuda()
        cap_inputs = batch_data['cap_targets'].cuda().long()
        noc_labels = batch_data['noc_labels'].cuda()
        noc_objs = batch_data['noc_objs'].long().cuda()
        feats_fc = batch_data['feats_fc'].cuda()
        if not opts.use_att:
            feats_att = None
        else:
            feats_att = batch_data['feats_att'].cuda()

        cap_outs, noc_outs, ppx_outs = model(feats_fc, feats_att, cap_inputs, noc_objs[1:, :],
                                             det_feats, det_values, is_train=False)

        _, cap_outs = torch.max(cap_outs, -1)
        cap_outs = cap_outs.detach().cpu().numpy()
        for b in range(cap_outs.shape[1]):
            first_cap = []
            for cid in cap_outs[:, b]:
                word = vocab[cid]
                if word == '<eos>':
                    break
                first_cap.append(word)
            names_wo_novel.append(vnames[b])
            results_wo_novel.append(' '.join(first_cap))
            
    score_log = open(os.path.join(opts.checkpoint, 'eval_log.txt'), 'a')

    # METEOR wo novel
    gt, res = {}, {}
    for i, vname in enumerate(names_wo_novel):
        gt[vname] = cap_gt[vname]
        res[vname] = [{'caption': results_wo_novel[i]}]
    meteor = metric.score(gt, res, opts, score_log, epoch, False)
    return meteor


def post_process(cap_outs, noc_cls, noc_outs, noc_scores, vocab):
    '''     Rule of change word --> novel object:
            1. perplexity: noc cls [1]>[0] or [1]>cls_thres
            2. limit of change num in a sentence : change_num
            3. semantic similarity > sem_thres
    '''
    cls_thres = 0.1
    change_num = 3

    sem_thres = 0.1
    vis_thres = 0.1
    top_k = 8

    first_cap, final_cap = [], []
    for cid in cap_outs:
        word = vocab[cid]
        if word == '<eos>':
            break
        first_cap.append(word)
        final_cap.append(word)

    first_cap = ' '.join(first_cap)
    final_cap = ' '.join(final_cap)

    return first_cap, final_cap


def revise_caption(batch_data, cap_outs, noc_cls, noc_outs,  # noc_sims,
                det_values, opts, vocab, vnames, show):
    first_caps = []
    final_caps = []
    _, cap_idxs = torch.max(cap_outs, -1)
    _, noc_idxs = torch.max(noc_outs, -1)

    seq_len, batch_len = cap_outs.shape[0], cap_outs.shape[1]
    for b in range(batch_len):
        vname = vnames[b]
        first_cap, final_cap = [], []
        for c, cid in enumerate(cap_idxs[:, b]):
            word = vocab[cid.detach().cpu().item()]
            first_cap.append(word)
            final_cap.append(word)

            if word == '<eos>':
                break

        # ---------- pick the top-k not sure
        top_k = 8
        if 2 in cap_idxs[:, b]:
            first_eos = cap_idxs[:, b].detach().cpu().numpy().tolist().index(2)
            noc_prob = noc_cls[:first_eos, b]
        else:
            noc_prob = noc_cls[:, b]
            first_eos = 20
        if first_eos < top_k:
            top_k = first_eos

        noc_prob /= max(noc_prob)

        ex_plcs = torch.topk(noc_prob.view(-1, ), top_k)[1]
        ex_plcs = ex_plcs.detach().cpu().numpy().tolist()

        det_objs = det_values[b, :].detach().cpu().numpy().tolist()
        for p in ex_plcs:
            prev_word = -1
            next_word = -1
            curr_word = vocab[cap_idxs[p, b].detach().cpu().item()]
            nid = noc_idxs[p, b].detach().cpu().item()

            # --------- skate board : skateboard
            if p > 0:
                prev_word = vocab[cap_idxs[p-1, b].detach().cpu().item()]
                if prev_word + curr_word in [vocab[nid], vocab[nid] + 's', vocab[nid] + 'es']:
                    continue
            if p < seq_len-1:
                next_word = vocab[cap_idxs[p+1, b].detach().cpu().item()]
                if curr_word + next_word in [vocab[nid], vocab[nid] + 's', vocab[nid] + 'es']:
                    continue

            # --------- exchange novel object
            # if nid in det_objs:
            if noc_prob[p] >= 0.1 and nid in det_objs:  # avoid repeat of same object
                #  person
                if nid == 0 and (first_cap[p] == 'man' or first_cap[p] == 'woman' or next_word == 'player'):
                    continue

                cid = nid + 9426
                novel_word = vocab[cid]
                if novel_word+'s' == curr_word or novel_word+'es' == curr_word:
                    continue
                # prevent stop stop sign / tennis tennis player
                split_word = novel_word.split()
                if len(split_word) > 1:
                    if split_word[0] == prev_word:
                        novel_word = split_word[1]
                    elif split_word[1] == next_word or split_word[1]+'s' == next_word or split_word[1]+'es' == next_word:
                        novel_word = split_word[0]

                final_cap[p] = novel_word
                det_objs.remove(nid)

        first_caps.append(' '.join(first_cap[:-1]))
        final_caps.append(' '.join(final_cap[:-1]))

    if show:
        utils.show(batch_data, vocab, final_caps[-1], cap_idxs[:, -1], noc_prob, noc_idxs[:, -1], vname[-1])
    return first_caps, final_caps


def generate_caption(batch_data, cap_outs, noc_outs, vocab, test_num):
    final_caps = []
    _, caps = torch.max(cap_outs, -1)
    _, nocs = torch.max(noc_outs, -1)
    seq_len, bch_len = caps.shape[0], caps.shape[1]

    for b in range(bch_len):
        final_cap = []
        for c, cid in enumerate(caps[:, b]):
            word = vocab[cid.detach().cpu().item()]
            final_cap.append(word)
            if word == '<eos>':
                break
        final_caps.append(' '.join(final_cap[:-1]))

        if random.uniform() < (6/test_num):
            print(final_caps[-1])
    return final_caps



