import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import numpy as np

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from utils.transforms import *

from models import *
from trainer import *
from datasets.dataset import *
from criterions import *
# from apex import amp


def getModel(args, device):

    modelDict = {
        'Multi_Phase_Lits': Multi_Phase_Lits,
        'Cross_Modal_Guidance': Cross_Modal_Guidance,
    }

    model = modelDict[args.arch](args.expand_size * 2 + 1, args.output_size)
    model = nn.DataParallel(model).to(device)

    return model

def getCriterion(args, device):
    
    if args.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'Dice':
        criterion = BinaryDiceLoss(p=1, smooth=1)
    elif args.criterion == 'Multi_Phase_Criterion':
        criterion = Multi_Phase_Criterion(device)
    elif args.criterion == 'CMG_Resnet_Criterion':
        criterion = CMG_Resnet_Criterion()
    
    
    return criterion

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        if torch.cuda.is_available():
            # torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            # cudnn.deterministic = True
            # cudnn.benchmark = False
    else:
        # cudnn.benchmark = True
        pass

    
    trainTransform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # RandomRotation([-30, 30]),
        ToTensor_LITS()
        ])
    valTransform = Compose([
        ToTensor_LITS()
    ])

    
    data_path = os.path.join(args.data_dir, args.dataset_name)
    data = split_data(data_path, nfolds=args.folds, expand_size=args.expand_size, random_state=args.seed)

    for fold in range(args.folds):
        # if fold in [0,1,2] :
        #     continue

        train_paths, val_paths = data[fold]

        trainset = labeledDataset(train_paths, trainTransform, train=True)
        valset = labeledDataset(val_paths, valTransform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valloader = DataLoader(valset, batch_size=args.batch_size * 2, shuffle=True, num_workers=args.workers)


        model = getModel(args, device)
        criterion = getCriterion(args, device)
        optimizer = optim.AdamW(model.parameters(), args.lr)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.patience, gamma=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, patience=args.patience)
        trainer = Trainer(args, model, criterion, optimizer, trainloader, valloader, device, scheduler=scheduler, fold=fold)
        trainer.train()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cross-modal-guidance")

    # path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(working_dir, 'results')
    datesets_dir = os.path.join(working_dir, 'datasets')

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=datesets_dir)
    parser.add_argument('--dataset-name', type=str, default='data/MPTH/train_img/pv')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'logs'))
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'weights'))
    parser.add_argument('--predicts-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'predicts'))
    parser.add_argument('--plots-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'plots'))

    # model
    parser.add_argument('--expand_size', type=int, default=0)
    parser.add_argument('-a', '--arch', type=str, default='Cross_Modal_Guidance')
    parser.add_argument('--output-size', type=int, default=1)
    # parser.add_argument('--dropout', type=float, default=0.2)

    # # criterion
    parser.add_argument('-c', '--criterion', type=str, default='MF_Resnet_Criterion_v2')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--patience', type=int, default=40, help="learning rate reduce patience")
    parser.add_argument('--accumulate-step', type=int, default=1)

    # training configs
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=12)
    parser.add_argument('-j', '--workers', type=int, default=8)

    args = parser.parse_args()
    main(args)