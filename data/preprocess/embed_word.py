from __future__ import print_function
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
import six
import os, sys
import numpy as np
import argparse
sys.path.append('../')
from misc.cap_metrics import create_emb_layer
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl


def get_vocabs(dict_file):
    with open(dict_file, "rb") as f:
        vocab = pkl.load(f)

    cap_vocab = vocab["lstm_vocab"]
    det_vocab = vocab["detcls_vocab"]

    return cap_vocab, det_vocab


def get_embeddings(file_enc, opt):
    embs = dict()

    for (i, l) in enumerate(open(file_enc, 'rb')):
        if not l:
            break
        if len(l) == 0:
            continue

        l_split = l.decode('utf8').strip().split(' ')
        if len(l_split) == 2:
            continue
        embs[l_split[0]] = [float(em) for em in l_split[1:]]
    print("Got {} decryption embeddings from {}".format(len(embs),
                                                                  file_enc))
    return embs


def match_embeddings(vocab, emb, opt):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}

    twin_pair = {}
    with open('/data/qfeng/coco/glove/novel_twin_pair.txt', 'r') as f:
        for line in f.readlines():
            k, v = line.split(':')[0], line.split(':')[1].split()[0]
            print(k, v)
            twin_pair[k] = v
    miss_words = []
    for w_id, w in enumerate(vocab):
        # if w in corr_pair.keys():
        #     wc = corr_pair[w]
        #     filtered_embeddings[w_id] = emb[wc]
        #     count['match'] += 1
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        elif len(w.split()) > 1:
            # emb_0 = emb[w.split()[0]]
            # emb_1 = emb[w.split()[1]]
            # combine_emb = []
            # for i in range(300):
            #     combine_emb.append((emb_0[i] + emb_1[i]) / 2)
            # filtered_embeddings[w_id] = combine_emb
            filtered_embeddings[w_id] = emb[twin_pair[w]]
            count['match'] += 1
        else:
            print(u"not found:\t{}".format(w))
            count['miss'] += 1
            miss_words.append(w)

    return torch.Tensor(filtered_embeddings), count, miss_words


def main(opt):
    cap_vocab, det_vocab = get_vocabs(opt.dict_file)
    total_vocab = cap_vocab + det_vocab
    embeddings = get_embeddings(opt.emb_file, opt)

    # cap_corr_path = '/data/qfeng/coco/glove/pair_miss_cap.txt'
    # det_corr_path = '/data/qfeng/coco/glove/pair_miss_det.txt'

    filtered_cap_embeddings, count, miss = match_embeddings(total_vocab, embeddings, opt)
    print('cap', count)
    torch.save(filtered_cap_embeddings, opt.output_file + "_cap_total.pt")

    # filtered_det_embeddings, count, miss = match_embeddings(det_vocab, embeddings, opt)
    # print('det', count)
    # torch.save(filtered_det_embeddings, opt.output_file + "_det.pt")

    print('\nDone!')


def find_similar_word(opt):
    # similar novel object pairs
    cap_vocab, det_vocab = get_vocabs(opt.dict_file)
    cap_words = []
    det_words = []
    for cid, cw in enumerate(cap_vocab):
        cap_words.append(cw)
    for did, dw in enumerate(det_vocab):
        det_words.append(dw)

    cap_embedder = create_emb_layer(opt.output_file + "_cap.pt")
    det_embedder = create_emb_layer(opt.output_file + "_det.pt")

    cap_ids = torch.Tensor(range(len(cap_vocab))).long().squeeze()
    cap_embeddings = cap_embedder(cap_ids)

    for det_id, det_word in enumerate(det_vocab):
        print('----', det_word)
        det_embedding = det_embedder(torch.Tensor([det_id]).long())
        det_embeddings = torch.stack([det_embedding]*len(cap_vocab)).squeeze()

        similarity = F.cosine_similarity(cap_embeddings, det_embeddings, -1)
        top_k = 5
        scores, neighbors = torch.topk(similarity.view(-1, ), top_k)

        for n in neighbors:
            print(cap_vocab[n])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    # parser.add_argument('-emb_file', required=True,
    #                     help="target Embeddings from this file")
    # parser.add_argument('-output_file', required=True,
    #                     help="Output file for the prepared data")
    # parser.add_argument('-dict_file', required=True,
    #                     help="Dictionary file")
    opt = parser.parse_args()
    opt.dict_file = "/data/qfeng/coco/noc/coco_split.vocab"
    opt.output_file = "/data/qfeng/coco/glove/42B/embeds"
    opt.emb_file = "/data/qfeng/coco/glove/42B/glove.42B.300d.txt"

    main(opt)

    # find_similar_word(opt)
