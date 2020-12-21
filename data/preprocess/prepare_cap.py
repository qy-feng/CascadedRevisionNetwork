import json
import os, sys
import nltk
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl


def get_vocab(data_dir):
    split_vocab_path = 'noc/coco_split.vocab'
    split_vocab_path = os.path.join(data_dir, split_vocab_path)
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
    return wordtoix, ixtoword


def ex_similar_pair(data_dir, subset):
    wordtoix, ixtoword = get_vocab(data_dir)

    ex_solo_pair = {}
    ex_solo_idx = []
    with open(os.path.join(data_dir, "glove/novel_solo_pair.txt"), 'r') as f:
        for line in f.readlines():
            det, cap = line.split(':')
            cap = cap.split()[0]
            ex_solo_idx.append(wordtoix[det])
            ex_solo_pair[wordtoix[det]] = wordtoix[cap]

    ex_twin_pair = {}
    ex_twin_idxs = {}  # second word:first word
    with open(os.path.join(data_dir, "glove/novel_twin_pair.txt"), 'r') as f:
        for line in f.readlines():
            det, cap = line.split(':')
            cap = cap.split()[0]
            idx0, idx1 = wordtoix[det.split()[0]], wordtoix[det.split()[1]]
            ex_twin_idxs[idx1] = idx0
            ex_twin_pair['%d_%d' % (idx0, idx1)] = wordtoix[cap]

    with open(os.path.join(data_dir, 'noc/cap_info_%s.pkl' % subset), 'rb') as f:
        text_data = pkl.load(f)

    new_cap_info = []
    for i, instance in enumerate(text_data):
        ori_cap = instance['caption_inputs']
        ori_nn = instance['nn']
        ex_twin_plcs = []
        for idx in range(len(ori_cap) - 1):
            if ori_cap[idx + 1] in ex_twin_idxs.keys():
                if ori_cap[idx] == ex_twin_idxs[ori_cap[idx + 1]]:
                    ex_twin_plcs.append(idx)

        new_captions = []
        new_nn = []
        novel_objects = []
        for idx in range(len(ori_cap)):
            if idx - 1 not in ex_twin_plcs:  # skip second word
                if idx in ex_twin_plcs:
                    cap_word = ixtoword[ori_cap[idx]] + ' ' + ixtoword[ori_cap[idx + 1]]
                    new_captions.append(ex_twin_pair['%d_%d' % (ori_cap[idx], ori_cap[idx + 1])])
                    new_nn.append(1)
                    novel_objects.append(wordtoix[cap_word])
                elif ori_cap[idx] in ex_solo_idx:
                    new_captions.append(ex_solo_pair[ori_cap[idx]])
                    new_nn.append(1)
                    novel_objects.append(ori_cap[idx])
                else:
                    new_captions.append(ori_cap[idx])
                    new_nn.append(ori_nn[idx])
                    novel_objects.append(0)

        instance['new_captions'] = new_captions
        instance['new_nn'] = new_nn
        instance['novel_objects'] = novel_objects
        new_cap_info.append(instance)

        if i % (len(text_data)/10) == 0:
            print(instance)

    with open(os.path.join(data_dir, 'noc/ex_cap_info_%s.pkl' % subset), 'wb') as f:
        pkl.dump(new_cap_info, f)


if __name__ == '__main__':

    data_dir = '/home/qianyu/data/qfeng/coco'
    subset = 'train'
    # subset = 'test'
    # ex_similar_pair(data_dir, subset)

    wordtoix, ixtoword = get_vocab(data_dir)

    no, so = {}, {}
    cnt_no = {}
    cnt_so = {}
    with open(os.path.join(data_dir, "glove/novel_solo_pair.txt"), 'r') as f:
        for line in f.readlines():
            det, cap = line.split(':')
            cap = cap.split()[0]
            no[wordtoix[det]] = det
            cnt_no[det] = 0
            so[wordtoix[det]] = cap
            cnt_so[cap] = 0

    with open(os.path.join(data_dir, 'noc/cap_info_%s.pkl' % subset), 'rb') as f:
        text_data = pkl.load(f)

    cnt_no = {'toilet': 6874, 'bicycle': 1427, 'kite': 4138, 'skis': 4106, 'carrot': 338, 'donut': 832, 'snowboard': 1665,
              'sandwich': 3337, 'motorcycle': 4698, 'oven': 1207, 'keyboard': 2024, 'scissors': 1288, 'airplane': 3707,
              'couch': 0, 'mouse': 1091, 'chair': 2967, 'boat': 3729, 'apple': 977, 'sheep': 3762, 'horse': 5600, 'cup': 1595,
              'tv': 1391, 'backpack': 366, 'toaster': 106, 'bowl': 3669, 'microwave': 0, 'bench': 5772, 'book': 1276,
              'elephant': 4400, 'orange': 2951, 'tie': 2832, 'bird': 3517, 'knife': 1077, 'pizza': 0, 'fork': 817,
              'frisbee': 5483, 'umbrella': 4795, 'banana': 1728, 'bus': 0, 'bear': 5664, 'vase': 3009, 'toothbrush': 604,
              'spoon': 484, 'giraffe': 4863, 'sink': 5270, 'handbag': 50, 'broccoli': 2262, 'refrigerator': 1934, 'laptop': 5265,
              'remote': 1242, 'surfboard': 3988, 'cow': 2147, 'car': 4284, 'clock': 6602, 'skateboard': 6825, 'dog': 10941,
              'bed': 7378, 'cat': 9485, 'person': 15352, 'train': 10085, 'truck': 4732, 'bottle': 0, 'suitcase': 0, 'cake': 5727,
              'zebra': 0}
    select_novels = []
    novels_images = {}
    for k, v in cnt_no.items():
        if v >= 1000:
            select_novels.append(k)
            novels_images[k] = []

    for i, instance in enumerate(text_data):
        caption = instance['caption_inputs']
        # for nk in no.keys():
        #     if nk in caption:
        #         cnt_no[no[nk]] += 1
        # for sk in so.keys():
        #     if sk in caption:
        #         cnt_so[so[sk]] += 1
        for novel in select_novels:
            idx = wordtoix[novel]
            if idx in caption:
                novels_images[novel].append(instance['vname'])
        print(novel, len(novels_images[novel]))

    with open(os.path.join(data_dir, 'noc/novel_images.pkl'), 'wb') as f:
        pkl.dump(novels_images, f)
