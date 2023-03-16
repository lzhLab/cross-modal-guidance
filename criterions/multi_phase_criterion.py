import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Multi_Phase_Criterion(nn.Module):

    def __init__(self, device=None):
        super(Multi_Phase_Criterion, self).__init__()
        self.device = device
    
    def forward(self, predicts, targets):
        initial_seg, final_seg = predicts

        targets[targets > 0] = 1.
        targets = targets.squeeze(1).long()
        
        loss1 = F.cross_entropy(initial_seg, targets)
        loss2 = F.cross_entropy(final_seg, targets)

        return (loss1 + loss2) / 2