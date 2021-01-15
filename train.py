import os
import argparse
import logging
import sys
import re

import torch
import torch.nn as nn
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

    # Environment
    os.makedirs(args.exp, exist_ok=True)
    logger = make_logger(name='exp',
                         save_dir=args.exp,
                         save_filename='train.log')

    # Dataset
    # train_transforms = transforms.Compose([
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize([0.5], [0.5])
    #                                        ])
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    train_set = Dataset(split_file=args.train_split, 
                        split='training', 
                        root=args.root_train, 
                        mode=args.mode, 
                        snippets=args.num_frames, 
                        transforms=train_transforms, 
                        num_classes=args.num_classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             num_workers=args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=True)

    # Checkpoint loading
    steps = 0
    if args.mode == 'rgb':
        in_channels = 3
        pretrained_default = torch.load('models/rgb_imagenet.pt')
    elif args.mode == 'flow':
        in_channels = 2
        pretrained_default = torch.load('models/flow_imagenet.pt')
    else:
        raise ValueError('Unknown mode %s. Only rgb or flow.' %args.mode)
    
    if args.resume is not None:
        print('Resuming {}...'.format(args.resume))
        pretrained_point = torch.load(args.resume)['state_dict']
        steps = int(re.findall(r'\d+', os.path.basename(args.resume))[0])
        # tot_losses = checkpoint['loss']
        # if dataset=='thumos':
        #     epoch = int(steps*args.snippets*args.batch_size / 1214016)
        # else:
        #     epoch = int(steps*args.snippets*args.batch_size / 5482688)
    
    # Model loading
    net = InceptionI3d(num_classes=400, 
                       in_channels=in_channels, 
                       use_crf=args.crf, 
                       num_updates_crf=args.num_updates_crf, 
                       pairwise_cond_crf=args.pairwise_cond_crf)
    net.load_state_dict(pretrained_default)
    net.replace_logits(args.num_classes)
    net = net.to(device)
    net = nn.DataParallel(net)
    # net.freeze_layers()

    # Training setup: optimiser, loss
    # Scheduler may not be necessary
    classification_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                              milestones=[1000],
                                              gamma=0.1)
    if steps > 0:
        for i in range(steps):
            lr_sched.step()

    tot_losses = 0
        
    # Training loop
    for epoch in range(args.max_epoch):
        for cnt, (imgs, labels) in enumerate(train_loader):
            steps += 1
            imgs = imgs.to(device)
            labels = labels.to(device)

            # obtain probability map from logits, and single label from label map
            probs = net(imgs)
            probs = probs.mean(2)
            probs_logits = probs
            probs = nn.Softmax(1)(probs_logits)
            labels = labels[:, :, 0].argmax(dim=1)
            cls_loss = classification_criterion(probs, labels)

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()

            tot_losses += cls_loss.item()
            avg_loss = tot_losses / steps

            info = 'Epoch {}/{} Iteration {}/{} Avg Loss: {:.4f} Cur Loss {:.4f}'.format(epoch,
                                                                         args.max_epoch,
                                                                         cnt,
                                                                         len(train_loader),
                                                                         avg_loss,
                                                                         cls_loss.item())
            logger.info(info)

            if (steps % args.save_every) == 0:
                logger.info('Saving to {}...'.format('steps{:07d}.pt'.format(steps)))
                ckpt_name = os.path.join(args.exp, 'steps{:07d}.pt'.format(steps))
                save_dict = {
                    'state_dict': net.state_dict()
                }
                torch.save(save_dict, ckpt_name)
            
            lr_sched.step()
            #import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--mode', type=str, help='rgb or flow', default='rgb')
    parser.add_argument('--exp', type=str, default='./experiments')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # Dataset args
    parser.add_argument('--dataset', help='multithumos or charades', type=str, default='multithumos')
    parser.add_argument('--root_train', type=str)
    parser.add_argument('--train_split', type=str)
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
