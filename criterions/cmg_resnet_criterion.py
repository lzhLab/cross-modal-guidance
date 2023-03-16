import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CMG_Resnet_Criterion(nn.Module):

    def __init__(self):
        super().__init__()

        self.bceloss = nn.BCEWithLogitsLoss()
    
    def forward(self, predicts, targets):
        pred_2, pred_3, pred_4 = predicts
        # pv_pred_1, art_pred_1 = pred_1
        pv_pred_2, art_pred_2 = pred_2
        pv_pred_3, art_pred_3 = pred_3
        pv_pred_4, art_pred_4 = pred_4
        pv_mask, art_mask = targets

        # pv_targets_1 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_1.size()[3])
        pv_targets_2 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_2.size()[3])
        pv_targets_3 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_3.size()[3])
        pv_targets_4 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_4.size()[3])

        # art_targets_1 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_1.size()[3])
        art_targets_2 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_2.size()[3])
        art_targets_3 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_3.size()[3])
        art_targets_4 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_4.size()[3])


        # pv_bceloss_1 = self.bceloss(pv_pred_1, pv_targets_1)
        pv_bceloss_2 = self.bceloss(pv_pred_2, pv_targets_2)
        pv_bceloss_3 = self.bceloss(pv_pred_3, pv_targets_3)
        pv_bceloss_4 = self.bceloss(pv_pred_4, pv_targets_4)
        # art_bceloss_1 = self.bceloss(art_pred_1, art_targets_1)
        art_bceloss_2 = self.bceloss(art_pred_2, art_targets_2)
        art_bceloss_3 = self.bceloss(art_pred_3, art_targets_3)
        art_bceloss_4 = self.bceloss(art_pred_4, art_targets_4)

        loss = pv_bceloss_2 + pv_bceloss_3 + pv_bceloss_4 +\
                art_bceloss_2 + art_bceloss_3 + art_bceloss_4
        return loss / 6

class MF_Resnet_Criterion_CE(nn.Module):

    def __init__(self):
        super().__init__()

        self.celoss = nn.CrossEntropyLoss()
    
    def forward(self, predicts, targets):
        pred_2, pred_3, pred_4 = predicts
        # pv_pred_1, art_pred_1 = pred_1
        pv_pred_2, art_pred_2 = pred_2
        pv_pred_3, art_pred_3 = pred_3
        pv_pred_4, art_pred_4 = pred_4
        pv_mask, art_mask = targets
        pv_mask = pv_mask.squeeze(1).float()
        art_mask = art_mask.squeeze(1).float()

        # pv_targets_1 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_1.size()[3])
        pv_targets_2 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_2.size()[3])
        pv_targets_3 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_3.size()[3])
        pv_targets_4 = pv_mask

        # art_targets_1 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_1.size()[3])
        art_targets_2 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_2.size()[3])
        art_targets_3 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_3.size()[3])
        art_targets_4 = art_mask

        # pv_bceloss_1 = self.bceloss(pv_pred_1, pv_targets_1)
        pv_celoss_2 = self.celoss(pv_pred_2, pv_targets_2.long())
        pv_celoss_3 = self.celoss(pv_pred_3, pv_targets_3.long())
        pv_celoss_4 = self.celoss(pv_pred_4, pv_targets_4.long())
        # art_bceloss_1 = self.bceloss(art_pred_1, art_targets_1)
        art_celoss_2 = self.celoss(art_pred_2, art_targets_2.long())
        art_celoss_3 = self.celoss(art_pred_3, art_targets_3.long())
        art_celoss_4 = self.celoss(art_pred_4, art_targets_4.long())

        loss = pv_celoss_2 + pv_celoss_3 + pv_celoss_4 +\
                art_celoss_2 + art_celoss_3 + art_celoss_4
        return loss / 6

class CMG_Resnet_CE_DICE(nn.Module):

    def __init__(self):
        super().__init__()

        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = BinaryDiceLoss()
    
    def forward(self, predicts, targets):
        pred_2, pred_3, pred_4 = predicts
        # pv_pred_1, art_pred_1 = pred_1
        pv_pred_2, art_pred_2 = pred_2
        pv_pred_3, art_pred_3 = pred_3
        pv_pred_4, art_pred_4 = pred_4
        pv_mask, art_mask = targets
        pv_mask = pv_mask.squeeze(1).float()
        art_mask = art_mask.squeeze(1).float()

        pv_targets_2 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_2.size()[3])
        pv_targets_3 = F.adaptive_max_pool2d(pv_mask, output_size=pv_pred_3.size()[3])
        pv_targets_4 = pv_mask

        art_targets_2 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_2.size()[3])
        art_targets_3 = F.adaptive_max_pool2d(art_mask, output_size=art_pred_3.size()[3])
        art_targets_4 = art_mask

        pv_celoss_2 = self.celoss(pv_pred_2, pv_targets_2.long())
        pv_celoss_3 = self.celoss(pv_pred_3, pv_targets_3.long())
        pv_celoss_4 = self.celoss(pv_pred_4, pv_targets_4.long())
        art_celoss_2 = self.celoss(art_pred_2, art_targets_2.long())
        art_celoss_3 = self.celoss(art_pred_3, art_targets_3.long())
        art_celoss_4 = self.celoss(art_pred_4, art_targets_4.long())

        
        celoss = (pv_celoss_2 + pv_celoss_3 + pv_celoss_4 +\
                art_celoss_2 + art_celoss_3 + art_celoss_4) / 6

        diceloss = 0.
        pv_pred_2 = F.softmax(pv_pred_2, dim=1)
        pv_pred_3 = F.softmax(pv_pred_3, dim=1)
        pv_pred_4 = F.softmax(pv_pred_4, dim=1)
        art_pred_2 = F.softmax(art_pred_2, dim=1)
        art_pred_3 = F.softmax(art_pred_3, dim=1)
        art_pred_4 = F.softmax(art_pred_4, dim=1)

        for i in range(0, pv_pred_2.size(1)):
            
            pv_dc_2 = (pv_targets_2 == i).float()
            pv_dc_3 = (pv_targets_3 == i).float()
            pv_dc_4 = (pv_targets_4 == i).float()

            art_dc_2 = (art_targets_2 == i).float()
            art_dc_3 = (art_targets_3 == i).float()
            art_dc_4 = (art_targets_4 == i).float()

            pv_dc_loss_2 = self.diceloss(pv_pred_2[:,i], pv_dc_2)
            pv_dc_loss_3 = self.diceloss(pv_pred_3[:,i], pv_dc_3)
            pv_dc_loss_4 = self.diceloss(pv_pred_4[:,i], pv_dc_4)

            art_dc_loss_2 = self.diceloss(art_pred_2[:,i], art_dc_2)
            art_dc_loss_3 = self.diceloss(art_pred_3[:,i], art_dc_3)
            art_dc_loss_4 = self.diceloss(art_pred_4[:,i], art_dc_4)

            dice_loss_temp = (pv_dc_loss_2 + pv_dc_loss_3 + pv_dc_loss_4 +\
                                art_dc_loss_2 + art_dc_loss_3 + art_dc_loss_4) / 6
            diceloss += dice_loss_temp
        diceloss = diceloss / (pv_pred_2.size(1))

        celoss_weight = 0.5
        
        # return (celoss + diceloss / (celoss / diceloss).detach()) / 2
        return celoss * celoss_weight + diceloss * (1 - celoss_weight)
        # return diceloss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        target = target > 0
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))




if __name__ == '__main__':

    # a = torch.ones(2, 32, 32)
    # b = torch.ones(2, 1, 64, 64)
    # d = torch.ones(2, 1, 512, 512)

    a = (torch.randn((3,3)) * 1.5).int()
    a[a < 0] = 0

    print(a)
    print(a.size())

    a = F.one_hot(a.long(), 3)
    print(torch.transpose(a,0,2))
    print(a.size())

    # predicts = (a, b, d)
    # targets = torch.ones(2, 2, 512, 512)

    # criterion = MF_Unet_Criterion()
    # criterion(predicts, targets)