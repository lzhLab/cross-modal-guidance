import argparse
import os
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from utils.transforms import *

from models import *
from trainer_chaos import *
from datasets.CHAOS_dataset import *
from criterions import *
# from apex import amp



def getModel(args, device):

    modelDict = {
        'RFNet': RFNet,
        'Multi_Scale_Attention_Net': Multi_Scale_Attention_Net,
        'DMFNet': DMFNet,
        'Cross_Modal_Guidance_CHAOS': Cross_Modal_Guidance_CHAOS,
        # 'TransUNet': TransUNet
    }

    model = modelDict[args.arch](args.expand_size * 2 + 1, args.output_size)
    model = nn.DataParallel(model).to(device)

    return model

def getCriterion(args, device):
    
    if args.criterion == 'Bce_Dice':
        criterion = BCE_and_Dice(args.dice_weight, args.pos_weight, smooth=1)
    elif args.criterion == 'BCEWithLogitsLoss':
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'Dice':
        criterion = DiceLoss()
    elif args.criterion == 'fpn_unet_criterion':
        criterion = FPN_Unet_Criterion()
    elif args.criterion == 'mf_unet_criterion':
        criterion = MF_Unet_Criterion()
    elif args.criterion == 'MSELoss':
        criterion = nn.MSELoss()
    elif args.criterion == 'BCELoss':
        criterion = nn.BCELoss()
    elif args.criterion == 'Multi_Phase_Criterion':
        criterion = Multi_Phase_Criterion(device)
    elif args.criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'WCELoss':
        criterion = WCELoss(args.pos_weight)
    elif args.criterion == 'BELoss':
        criterion = BELoss()
    elif args.criterion == 'mf_resnet_criterion':
        criterion = MF_Resnet_Criterion()
    elif args.criterion == 'ce_local_dice_loss':
        criterion = CE_Local_Dice_Loss()
    elif args.criterion == 'semi_supervised_criterion':
        criterion = Semi_Supervised_Criterion()
    elif args.criterion == 'MF_Resnet_bce_dice':
        criterion = MF_Resnet_bce_dice()
    elif args.criterion == 'MF_Resnet_Criterion_v2':
        criterion = MF_Resnet_Criterion_v2()
    elif args.criterion == 'MF_Resnet_Criterion_v3':
        criterion = MF_Resnet_Criterion_v3()
    elif args.criterion == 'MSA_Critierion':
        criterion = MSA_Critierion()
    elif args.criterion == 'RFNet_Criterion':
        criterion = RFNet_Criterion()
    elif args.criterion == 'MF_Resnet_Criterion_CE':
        criterion = MF_Resnet_Criterion_CE()
    elif args.criterion == 'MF_Resnet_CE_DICE':
        criterion = MF_Resnet_CE_DICE()
    elif args.criterion == 'CE_and_Dice':
        criterion = CE_and_Dice()
    
    
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
        RandomRotation([-15, 15]),
        CenterCrop(256),
        ToTensor(),
        # Normalize([0.], [1.])
        Normalize_2d([0.], [1.])
        ])
    valTransform = Compose([
        CenterCrop(256),
        ToTensor(),
        # Normalize([0.], [1.])
        Normalize_2d([0.], [1.])
    ])
    
    data_path = os.path.join(args.data_dir, args.dataset_name)
    data = split_data(data_path, nfolds=args.folds, expand_size=args.expand_size, random_state=args.seed)
    # data = split_data_folder(data_path, nfolds=args.folds, expand_size=args.expand_size, random_state=args.seed)

    for fold in range(args.folds):
        # if fold in [0] :
        #     continue

        train_paths, val_paths = data[fold]

        trainset = chaosDataset(train_paths, trainTransform)
        # valset = chaosDataset(val_paths, valTransform)
        # trainset = VolumnCHAOSDataset(train_paths, trainTransform, True, size=(16, 128, 128))
        valset = VolumnCHAOSDataset(val_paths, valTransform, False)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=args.workers)


        model = getModel(args, device)
        criterion = getCriterion(args, device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
        optimizer = optim.AdamW(model.parameters(), args.lr)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.patience, gamma=0.5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=args.patience)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, patience=10)
        trainer = Trainer(args, model, criterion, optimizer, trainloader, valloader, device, scheduler=scheduler, fold=fold, expand_size=args.expand_size)
        trainer.train()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cross-modal-guidance")

    # path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(working_dir, 'results')
    datesets_dir = os.path.join(working_dir, 'datasets')

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=datesets_dir)
    parser.add_argument('--dataset-name', type=str, default='data/CHAOS/T1/image')
    # parser.add_argument('--dataset-name', type=str, default='data/CHAOS_T1_in_out/inphase/image')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'logs'))
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'weights'))
    parser.add_argument('--predicts-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'predicts_chaos'))
    parser.add_argument('--plots-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'plots'))

    # training configs
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100, help="learning rate reduce patience")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--accumulate-step', type=int, default=1)

    # model
    parser.add_argument('--expand_size', type=int, default=0)
    parser.add_argument('--output-size', type=int, default=5)

    # parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('-a', '--arch', type=str, default='Cross_Modal_Guidance_CHAOS')

    # # criterion
    parser.add_argument('-c', '--criterion', type=str, default='CMG_Resnet_CE_DICE')
    parser.add_argument('--dice-weight', type=int, default=1, help="weight of criterion dice loss, bce default 1")

    args = parser.parse_args()
    main(args)