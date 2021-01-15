import os
import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import videotransforms

from pytorch_i3d import InceptionI3d


def make_logger(name, save_dir, save_filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename+".txt"), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def train(args):
    if args.device:
        device = torch.device('cuda:{}'.format(args.device))
    else:
        device = torch.device('cuda')

    # Dataset
    # test_transforms = transforms.Compose([
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([0.5], [0.5])
    #                                        ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    val_dataset = Dataset(split_file=args.eval_split, 
                        split='testing', 
                        root=args.root_eval, 
                        mode=args.mode, 
                        snippets=args.num_frames, 
                        transforms=test_transforms, 
                        num_classes=args.num_classes)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             num_workers=args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=True)

    # Checkpoint loading
    # steps = 0
    # if args.mode == 'rgb':
    #     in_channels = 3
        # pretrained_default = torch.load('models/rgb_imagenet.pt')
    # elif args.mode == 'flow':
    #     in_channels = 2
        # pretrained_default = torch.load('models/flow_imagenet.pt')
    # else:
    #     raise ValueError('Unknown mode %s. Only rgb or flow.' %args.mode)
    
    # if args.resume is not None:
    #     print('Resuming {}...'.format(args.resume))
    #     pretrained_point = torch.load(args.resume)
    #     pretrained_point = checkpoint['state_dict']
    #     tot_losses = checkpoint['loss']
    #     steps = int(args.resume[:-3])
    #     if dataset=='thumos':
    #         epoch = int(steps*args.snippets*args.batch_size / 1214016)
    #     else:
    #         epoch = int(steps*args.snippets*args.batch_size / 5482688)
    
    # Model loading
    if args.mode == 'rgb':
        in_channels = 3
    elif args.mode == 'flow':
        in_channels = 2
    else:
        raise ValueError('Unknown mode %s. Only rgb or flow.' %args.mode)
    
    net = InceptionI3d(num_classes=400, 
                       in_channels=in_channels, 
                       use_crf=args.crf, 
                       num_updates_crf=args.num_updates_crf, 
                       pairwise_cond_crf=args.pairwise_cond_crf)
    net.replace_logits(args.num_classes)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(args.ckpt)['state_dict'], strict=False)
    for param in net.parameters():
        param.requires_grad = False
    net = net.to(device)
    net.eval()

    # Validation setup: optimiser, loss
    probs = []
    preds = []
    anns = []
        
    #import ipdb; ipdb.set_trace()

    with torch.no_grad():
        for cnt, (imgs, labels) in enumerate(val_loader):
            print('Evaluating in progress {}/{}...'.format(cnt, len(val_loader)))
            imgs = imgs.to(device)

            prob = net(imgs)
            prob = prob.mean(2)
            prob_logits = prob
            prob = nn.Softmax(1)(prob_logits)

            pred = F.softmax(prob, dim = 1)
            pred = pred.argmax(-1)

            labels = labels[:, :, 0].argmax(dim=1)
            
            probs.append(prob)
            preds.append(pred)
            anns.append(labels)
            
        
        anns = torch.cat(anns)
        probs = torch.cat(probs).cpu()
        preds = torch.cat(preds).cpu()
        torch.save(probs, args.ckpt.replace('.pt', '.predictions.pt'))
        torch.save(anns, args.ckpt.replace('.pt', '.labels.pt'))
        try:
            correct = int((anns == preds).sum())
        except:
            import ipdb; ipdb.set_trace()
        acc = correct/len(preds)
    print('Acc is {}'.format(acc))
    #print('Accuraccy for epoch {} iteration {} is {}/{}'.format(epoch, cnt, acc, correct))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--mode', type=str, help='rgb or flow', default='rgb')
    parser.add_argument('--exp', type=str, default='./experiments')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    
    # Dataset args
    parser.add_argument('--dataset', help='multithumos or charades', type=str, default='multithumos')
    parser.add_argument('--root_eval', type=str)
    parser.add_argument('--eval_split', type=str)
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=155)
    
    # Training args
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)

    # CRF args
    parser.add_argument('-crf', type=bool, default=False)
    parser.add_argument('-num_updates_crf', type=int, default=1)
    parser.add_argument('-pairwise_cond_crf', type=bool, default=False)
    parser.add_argument('-reg_crf', type=float, default=-1)
    parser.add_argument('-reg_type', type=str, default='l2')

    args = parser.parse_args()

    if args.dataset=='multithumos':
        from datasets.multithumos_dataset import Multithumos as Dataset
    elif args.dataset=='charades':
        from datasets.charades_dataset import Charades as Dataset
    elif args.dataset=='uavhuman':
        from datasets.uavhuman_dataset import Uavhuman as Dataset
    
    train(args)
