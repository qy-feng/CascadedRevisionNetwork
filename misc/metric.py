# from __future__ import print_function, division, absolute_import
import os, sys
sys.path.append("../coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from math import *


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def pack_crit(result, target, lengths, crit='C', weight=None):
    result = pack_padded_sequence(result, lengths).data.squeeze()
    target = pack_padded_sequence(target, lengths).data
    if crit == 'B':
        loss = F.binary_cross_entropy(result, target.float(), weight)
    else:
        loss = F.cross_entropy(result, target.long(), weight)
    return loss


def mask_crit(result, target, masks, crit='C'):
    non_zeros = torch.nonzero(masks.view(-1, )).squeeze().cuda()
    if non_zeros.sum() == 0:
        return torch.tensor([0]).float().cuda()
    else:
        result = torch.index_select(result.view(-1, result.size(-1)), 0, non_zeros).squeeze()
        target = torch.index_select(target.view(-1, ), 0, non_zeros)
        if crit == 'B':
            loss = F.binary_cross_entropy(result, target.float())
        else:
            loss = F.cross_entropy(result, target.long())
        return loss


def cal_similarity(caps, nocs, cap_embedder, det_embedder):
    caps = torch.Tensor(caps).long().cuda()
    # print('cap', caps.shape)
    nocs = torch.Tensor(nocs).long().cuda()
    # print('noc', nocs.shape)
    cap_embeds = cap_embedder(caps)
    noc_embeds = det_embedder(nocs)

    similarity = F.cosine_similarity(cap_embeds, noc_embeds, -1)

    return similarity


def create_emb_layer(emb_cap_path):
    emb_weight = torch.load(emb_cap_path)
    num_embed, embed_dim = emb_weight.size()
    print('pretrain embeddings:', num_embed, embed_dim)
    emb_layer = nn.Embedding(num_embed, embed_dim)
    emb_layer.load_state_dict({'weight': emb_weight})

    return emb_layer


def score(gts, res, opts, log_out, epoch, cal_F1=True):
    origingts = gts
    originres = res
    print>> log_out, opts.resume + "\n"
    print>> log_out, 'Epoch'+str(epoch) + "\n"
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")]
    """
    scorers = [(Meteor(), "METEOR")]
    for scorer, method in scorers:
        # print 'computing %s score...'%(scorer.method())
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print>> log_out, "%s: %f" % (m, sc)
                print("%s: %f" % (m, sc))
        else:
            print>> log_out, "%s: %f" % (method, score)
            print "%s: %f" % (method, score)
    if cal_F1:
        F1_score = F1(originres, origingts)
        avg = 0.0
        for noc_word in sorted(F1_score.keys()):
            print>> log_out, noc_word, F1_score[noc_word]
            print noc_word, F1_score[noc_word]
            avg += F1_score[noc_word]

        avg = avg / len(F1_score.keys())
        print "AVG", avg
        print>> log_out, "AVG", avg
        print>> log_out, "\n\n\n\n\n"

        return score, avg
    else:
        return score


def load_captions(generated_caption, caption_meta_path, stage, meta):
    # meta = pkl.load(open(caption_meta_path))
    gt, res = {}, {}
    for v in meta[stage]:
        vname = v['vname']
        if vname not in gt.keys():
            gt[vname] = []
        gt[vname].append({'caption': v['desc']})

    for vname, caption in generated_caption.iteritems():
        res[vname] = [{"caption": " ".join(caption)}]
    return gt, res


noc_objects = ['bus', 'bottle', 'couch', 'microwave', 'pizza', 'tennis_racket', 'suitcase', 'zebra']
# noc_objects = ['bear', 'cat', 'dog', 'elephant', 'horse', 'motorcycle']
# noc_objects = ['bed', 'book', 'carrot', 'elephant', 'spoon', 'toilet', 'truck', 'umbrella']


def split_sent(sent):
    sent = sent.lower()
    sent = re.sub('[^A-Za-z0-9\s]+', '', sent)
    sent = sent.replace('tennis racket', 'tennis_racket')
    return sent.split()


def F1(generated_caption, gt):
    F1_score = {}

    for noc_word in noc_objects:

        novel_images = []
        nonNovel_images = []
        for vname in gt.keys():
            has_novel = False
            for caption in gt[vname]:
                if noc_word in split_sent(caption["caption"]):
                    has_novel = True
                    break

            if has_novel:
                novel_images.append(vname)
            else:
                nonNovel_images.append(vname)

        # true positive are sentences that contain match words and should
        tp = sum([1 for name in novel_images if noc_word in split_sent(generated_caption[name][0]["caption"])])
        # false positive are sentences that contain match words and should not
        fp = sum([1 for name in nonNovel_images if noc_word in split_sent(generated_caption[name][0]["caption"])])
        # false nagative are sentences that do not contain match words and should
        fn = sum([1 for name in novel_images if noc_word not in split_sent(generated_caption[name][0]["caption"])])

        if noc_word.find("racket") != -1:
            print(noc_word, 'tp', tp, 'fp', fp, 'fn', fn)

        # precision = tp/(tp+fp)
        if tp > 0:
            precision = float(tp) / (tp + fp)
            # recall = tp/(tp+fn)
            recall = float(tp) / (tp + fn)
            # f1 = 2* (precision*recall)/(precision+recall)
            F1_score[noc_word] = 2.0 * (precision * recall) / (precision + recall)
        else:
            F1_score[noc_word] = 0.
    return F1_score


if __name__ == "__main__":
    import os
    data_dir = '/data/qfeng/coco'
    split_vocab_pkl = 'noc/coco_split.vocab'
    split_vocab_path = os.path.join(data_dir, split_vocab_pkl)
    with open(split_vocab_path, "rb") as f:
        vocab = pkl.load(f)

    det_vocab = vocab["detcls_vocab"]
    cap_vocab = vocab["lstm_vocab"]
    total_vocab = cap_vocab + det_vocab

    wordtoix = {}
    ixtoword = {}
    for idx, word in enumerate(total_vocab):
        wordtoix[word] = idx
        ixtoword[idx] = word

    caption = "kitten luggage"
    words = caption.split()
    idxs = []
    for word in words:
        idxs.append(wordtoix[word])

    nocs = [wordtoix['suitcase']-9426]*2
    # nocs = [wordtoix['cat']-9426] * 2
    print(idxs, nocs)
    idxs = torch.Tensor(idxs).long().cuda()
    nocs = torch.Tensor(nocs).long().cuda()
    print(cal_similarity(idxs, nocs, data_dir))
