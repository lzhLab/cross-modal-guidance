import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MSA_Critierion(nn.Module):

    def __init__(self):
        super().__init__()
        self.CE_loss = nn.CrossEntropyLoss()
        self.mseLoss = nn.MSELoss()
    
    def forward(self, output, target):
        semVector_1_1, \
        semVector_2_1, \
        semVector_1_2, \
        semVector_2_2, \
        semVector_1_3, \
        semVector_2_3, \
        semVector_1_4, \
        semVector_2_4, \
        inp_enc0, \
        inp_enc1, \
        inp_enc2, \
        inp_enc3, \
        inp_enc4, \
        inp_enc5, \
        inp_enc6, \
        inp_enc7, \
        out_enc0, \
        out_enc1, \
        out_enc2, \
        out_enc3, \
        out_enc4, \
        out_enc5, \
        out_enc6, \
        out_enc7, \
        outputs0, \
        outputs1, \
        outputs2, \
        outputs3, \
        outputs0_2, \
        outputs1_2, \
        outputs2_2, \
        outputs3_2 = output

        # Segmentation_class = self.getTargetSegmentation(target)
        Segmentation_class = target.squeeze(1).long()

        # Cross-entropy loss
        loss0 = self.CE_loss(outputs0, Segmentation_class)
        loss1 = self.CE_loss(outputs1, Segmentation_class)
        loss2 = self.CE_loss(outputs2, Segmentation_class)
        loss3 = self.CE_loss(outputs3, Segmentation_class)
        loss0_2 = self.CE_loss(outputs0_2, Segmentation_class)
        loss1_2 = self.CE_loss(outputs1_2, Segmentation_class)
        loss2_2 = self.CE_loss(outputs2_2, Segmentation_class)
        loss3_2 = self.CE_loss(outputs3_2, Segmentation_class)

        lossSemantic1 = self.mseLoss(semVector_1_1, semVector_2_1)
        lossSemantic2 = self.mseLoss(semVector_1_2, semVector_2_2)
        lossSemantic3 = self.mseLoss(semVector_1_3, semVector_2_3)
        lossSemantic4 = self.mseLoss(semVector_1_4, semVector_2_4)

        lossRec0 = self.mseLoss(inp_enc0, out_enc0)
        lossRec1 = self.mseLoss(inp_enc1, out_enc1)
        lossRec2 = self.mseLoss(inp_enc2, out_enc2)
        lossRec3 = self.mseLoss(inp_enc3, out_enc3)
        lossRec4 = self.mseLoss(inp_enc4, out_enc4)
        lossRec5 = self.mseLoss(inp_enc5, out_enc5)
        lossRec6 = self.mseLoss(inp_enc6, out_enc6)
        lossRec7 = self.mseLoss(inp_enc7, out_enc7)

        lossG = (loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2)\
            + 0.25 * (lossSemantic1 + lossSemantic2 + lossSemantic3 + lossSemantic4) \
            + 0.1 * (lossRec0 + lossRec1 + lossRec2 + lossRec3 + lossRec4 + lossRec5 + lossRec6 + lossRec7)  # CE_loss
        return lossG

    def getTargetSegmentation(self, batch):
        # input is 1-channel of values between 0 and 1
        # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
        # output is 1 channel of discrete values : 0, 1, 2 and 3
        
        denom = 0.24705882 # for Chaos MRI  Dataset this value

        return (batch / denom).round().long().squeeze()