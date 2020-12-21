from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from misc.metrics import cal_similarity
from random import randrange as rrange
from copy import copy
from numpy import random
from time import time

coco_known = [['bicycle','car','motorcycle','airplane','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench'],
              ['bird','cat','dog','horse','sheep','cow','elephant','bear','giraffe'],
              ['backpack','umbrella','handbag','tie'],
              ['frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard'],
              ['wine glass','cup','fork','knife','spoon','bowl'],
              ['banana','apple','sandwich','orange','broccoli','carrot','hot dog','donut','cake'],
              ['chair','potted plant','bed','dining table','toilet'],
              ['tv','laptop','mouse','remote','keyboard','cell phone'],
              ['oven','toaster','sink','refrigerator'],
              ['book','clock','vase','scissors','teddy bear','hair drier','toothbrush']]
coco_unknown = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'tennis racket', 'suitcase', 'zebra']


def get_pseudo_obj(ori_idx, i2w, w2i):
    ori_obj = i2w[ori_idx]
    psd_objs = []
    if ori_obj not in coco_unknown:
        for cat in range(len(coco_known)):
            if ori_obj in coco_known[cat]:
                psd_list = copy(coco_known[cat])
                psd_list.remove(ori_obj)
                psd_num = 1
                for _ in range(psd_num):
                    psd_obj = psd_list[rrange(len(psd_list))]
                    psd_objs.append(w2i[psd_obj])
                break
    if len(psd_objs) == 0:
        psd_objs.append(-1)
        # print(ori_idx, ori_obj)
    return psd_objs[0]


def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def NLLLoss(logits, targets, masks):
    bs, seq_len = logits.shape[1], logits.shape[0]
    out = torch.zeros((bs, seq_len), dtype=torch.float).cuda()

    logits = nn.functional.log_softmax(logits, dim=-1)
    norm = 0
    for i in range(bs):
        mask = torch.reshape(masks[:, i], (1, -1))
        logit = logits[:, i].permute(1, 0)[targets[:, i]]
        lo = torch.matmul(mask, logit)
        lo = torch.squeeze(lo)
        out[i, :] = lo/(torch.sum(mask) + 1)
        norm += torch.sum(mask)
    return -out.sum()/norm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma, multiple):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        if epoch < 0:
            lr *= 10
        else:
            lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


'''
----------------- Ablations -----------------
I (only caption) II (perplexity)
III (vis-match)  IV (sem-match)

0: I 
1: I + II + random select from detection
2: I + random select 2 pos + III + IV
3: I + II + III [old version]
4: I + II + IV 
5: full stages
---------------------------------------------
'''


def noc_matching(cap_outs, noc_cls, noc_outs, noc_sims, det_values,
                 opts, vocab, vnames, cap_embedder, det_embedder):
    first_caps = []
    final_caps = []
    seq_len, batch_len = cap_outs.shape[0], cap_outs.shape[1]
    _, cap_idxs = torch.max(cap_outs, -1)
    # vis-match objects
    _, noc_idxs = torch.max(noc_outs, -1)

    abl = opts.abl
    tic = time()
    for b in range(batch_len):
        ppl_thres = 0.15
        first_cap, final_cap = [], []
        sub_plcs = []
        sub_idxs = []
        cdd_idxs = []
        det_objs = det_values[b, :]
        noc_prob = noc_cls[:, b]/max(noc_cls[:, b])
        
        for c, c_idx in enumerate(cap_idxs[:, b]):
            # ---------------------- cap ---------------------- #
            word = vocab[c_idx.detach().cpu().item()]
            if word == '<eos>':
                break
            first_cap.append(word)
            final_cap.append(word)
            # ---------------------- noc ---------------------- #
            n_idx = noc_idxs[c, b]
            # 2: random select two position
            if abl == 2:
                if random.uniform() < 0.2:
                    sub_plcs.append(c)
                    sub_idxs.append(c_idx.detach().cpu().item())
                    cdd_idxs.append(n_idx.detach().cpu().item())
            elif abl == 4:
                sub_plcs.append(c)
                sub_idxs.append(c_idx.detach().cpu().item())
            elif noc_prob[c] >= ppl_thres:
                # 1: random select detected object
                if abl == 1:
                    o_idx = det_objs[random.randint(0, len(det_objs))]
                    final_cap[-1] = vocab[o_idx.detach().cpu().item()+9426]
                elif abl == 3:
                    final_cap[-1] = vocab[n_idx.detach().cpu().item()+9426]
                else:
                    sub_plcs.append(c)
                    sub_idxs.append(c_idx.detach().cpu().item())
                    cdd_idxs.append(n_idx.detach().cpu().item())

        if abl == 4:
            for n_idx in noc_idxs[:, b]:
                cdd_idxs.append(n_idx.detach().cpu().item())


        det_objs = det_objs.detach().cpu().numpy().tolist()

        if abl in [2, 4, 5]:
            # get each candidate's similarity with all substitutes
            if len(sub_idxs) > 0:
                for cdd_idx in cdd_idxs:
                    cdd_sims = cal_similarity(sub_idxs, [cdd_idx]*len(sub_idxs), cap_embedder, det_embedder)
                    max_sim, max_pos = torch.max(cdd_sims.unsqueeze(0), -1)

                    sub_plc = sub_plcs[max_pos.detach().cpu().item()]
                    det_obj = vocab[cdd_idx + 9426]

                    if max_sim > 0.25:
                    # if cdd_idx in det_objs:
                        if align(final_cap, sub_plc, det_obj):
                            final_cap[sub_plc] = det_obj
                            # det_objs.remove(cdd_idx)

        first_caps.append(' '.join(first_cap))
        final_caps.append(' '.join(final_cap))

    return first_caps, final_caps


def show(batch_data, vocab, sentence, caps, noc_cls, nocs, vname):
    ori_idx = batch_data['ori_caps'][1:, -1]
    tgt_idx = batch_data['cap_targets'][1:, -1]
    ori, tgt, res = [], [], []
    no = []
    for i in range(len(caps)):
        ori.append(vocab[ori_idx[i].detach().cpu().item()])
        tgt.append(vocab[tgt_idx[i].detach().cpu().item()])
        res.append(vocab[caps[i].detach().cpu().item()])
        no.append(vocab[nocs[i].detach().cpu().item() + 9426])

    print('ORI: ', ' '.join(ori))
    print('TGT: ', ' '.join(tgt))
    print('CAP: ', ' '.join(res))
    if noc_cls is not None:
        noc_cls = ' '.join([str(round(p, 2)) for p in noc_cls.detach().cpu().numpy()])
        print('noc cls:', noc_cls)
    print('noc: ', ' '.join(no))
    print('FIN: ', sentence + '\n')


def align(caps, pos, obj):
    # caps: original sentence
    # pos: position to change obj
    # obj: obj to put in sentence
    curr_word = caps[pos]
    sp_obj = obj.split()
    if pos > 0:
        prev_word = caps[pos-1]
        if len(sp_obj) > 1:
            if sp_obj[0] == prev_word:
                obj = sp_obj[1]
    else:
        prev_word = None

    if pos < len(caps)-1:
        next_word = caps[pos+1]
        if len(sp_obj) > 1:
            if sp_obj[1] == next_word:
                obj = sp_obj[0]
    else:
        next_word = None

    # zebras : zebra
    if curr_word in [obj + 's', obj + 'es']:
        return False

    # skate board: skateboard
    # stop sign: stop sign

    if prev_word is not None:
        if prev_word + ' ' + curr_word in [obj, obj + 's', obj + 'es']:
            return False

    if next_word is not None:
        if curr_word + ' ' + next_word in [obj, obj + 's', obj + 'es']:
            return False

    return True