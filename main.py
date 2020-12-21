from __future__ import print_function, division, absolute_import
import os
import time
import torch
from config import *
from data.dataloader import VisualText
import model
from train import train, evaluate
from misc import metric, utils


def main(opts):
    torch.manual_seed(123)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)

    tic = time.time()
    train_dataset = VisualText(opts, subset=opts.subset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size,
                                               shuffle=True, num_workers=opts.workers,
                                               pin_memory=True, collate_fn=train_dataset.my_collate_fn)
    print('time to load train:', time.time()-tic)
    tic = time.time()
    eval_dataset = VisualText(opts, is_train=False, subset=opts.subset)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=opts.batch_size,
                                              shuffle=False, num_workers=opts.workers,
                                              collate_fn=eval_dataset.my_collate_fn)
    print('time to load test:', time.time() - tic)
    vocab = eval_dataset.get_vocab()
    cap_gt = eval_dataset.get_cap()

    cap_model = model.Captioning(opts).cuda()
    params, multiple = get_parameters(cap_model, opts)
    if opts.optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr=opts.lr, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=opts.lr, weight_decay=opts.weight_decay)
    if opts.resume:
        cap_model, optimizer = load_model(opts, cap_model, optimizer)

    if opts.evaluate:
        evaluate(eval_loader, cap_model, opts, vocab, cap_gt, 0)

    lr = opts.lr
    best_score = 0
    for epoch in range(opts.start_epoch, opts.epochs):
        lr = utils.adjust_learning_rate(optimizer, epoch, lr, opts.schedule, opts.gamma, multiple=multiple)
        print('\nEpoch: %d | LR: %.8f' % (epoch, lr))

        train(train_loader, cap_model, optimizer, opts, epoch)

        if epoch > 0 and epoch % 2 == 0:
            meteor = evaluate(eval_loader, cap_model, opts, vocab, cap_gt, epoch)

            if best_score < meteor:
                best_score = meteor
                best_epoch = epoch
            print('BEST: ', best_score, '/', best_epoch)


def load_model(opts, model, optimizer):
    checkpoint = torch.load(opts.resume)
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    for k in pretrain_dict.keys():
        if k not in model_dict.keys():
            print(k)
    load_dict = {k: pretrain_dict[k] for k in model_dict.keys() if k in pretrain_dict.keys()}

    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if not opts.evaluate:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


def get_parameters(model, opts, default=True):
    if default:
        params = filter(lambda p: p.requires_grad, model.parameters())
        return params, [1.]
    else:
        lr_1 = []
        lr_2 = []
        params_dict = dict(model.named_parameters())
        for key, value in params_dict.items():
            if 'noc_cls_logit' in key:
                lr_1.append(value)
        params = [{'params': lr_1, 'lr': opts.lr},
                  {'params': lr_2, 'lr': opts.lr * 0.2}]

        return params, [1., 0.2]


if __name__ == '__main__':

    opts = parse_opts()

    if opts.machine == 'uts':
        opts.data_dir = '/home/qianyu/coco'

    opts.checkpoint = os.path.join('./checkpoint', opts.checkpoint)
    if not os.path.exists(opts.checkpoint):
        os.makedirs(opts.checkpoint)

    if opts.resume:
        opts.resume = '%s/%s/model_epoch_%d.pth.tar' % (opts.check_dir, opts.resume, opts.start_epoch - 1)
        print('\n Resume', opts.resume)

    if opts.evaluate:
        opts.resume = '%s/model_epoch_%d.pth.tar' % (opts.checkpoint, opts.start_epoch)
        print('\n Evaluate', opts.resume)

    print(opts)
    main(opts)
