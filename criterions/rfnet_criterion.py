import torch
import torch.nn as nn

class RFNet_Criterion(nn.Module):

    def __init__(self, ):
        super().__init__()
    
    def forward(self, output, target):
        target = torch.cat([target == 0, target == 1, target == 2, target == 3, target == 4], dim=1)
        fuse_pred, sep_preds, prm_preds = output

        fuse_cross_loss = self.softmax_weighted_loss(fuse_pred, target)
        fuse_dice_loss = self.dice_loss(fuse_pred, target)
        fuse_loss = fuse_cross_loss + fuse_dice_loss

        sep_cross_loss = torch.zeros(1).cuda().float()
        sep_dice_loss = torch.zeros(1).cuda().float()
        for sep_pred in sep_preds:
            sep_cross_loss += self.softmax_weighted_loss(sep_pred, target)
            sep_dice_loss += self.dice_loss(sep_pred, target)
        sep_loss = sep_cross_loss + sep_dice_loss

        prm_cross_loss = torch.zeros(1).cuda().float()
        prm_dice_loss = torch.zeros(1).cuda().float()
        for prm_pred in prm_preds:
            prm_cross_loss += self.softmax_weighted_loss(prm_pred, target)
            prm_dice_loss += self.dice_loss(prm_pred, target)
        prm_loss = prm_cross_loss + prm_dice_loss

        loss = fuse_loss + sep_loss + prm_loss

        return loss
    
    def softmax_weighted_loss(self, output, target, num_cls=5):
        target = target.float()
        B, _, H, W, Z = output.size()
        for i in range(num_cls):
            outputi = output[:, i, :, :, :]
            targeti = target[:, i, :, :, :]
            weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
            weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
            if i == 0:
                cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
            else:
                cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        cross_loss = torch.mean(cross_loss)
        return cross_loss
    
    
    def dice_loss(self, output, target, num_cls=5, eps=1e-7):
        target = target.float()
        for i in range(num_cls):
            num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
            l = torch.sum(output[:,i,:,:,:])
            r = torch.sum(target[:,i,:,:,:])
            if i == 0:
                dice = 2.0 * num / (l+r+eps)
            else:
                dice += 2.0 * num / (l+r+eps)
        return 1.0 - 1.0 * dice / num_cls