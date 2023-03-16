import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function


class _SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class _MaxPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gap = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self, input):
        size = input.size()[2:]
        pool = self.gap(input)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

    
class _MultiScaleConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Multi_Scale_Fusion_Block(nn.Module):

    def __init__(self, in_ch, fusion=False):
        super().__init__()
        if fusion:   
            self.fusion_dowmsample = nn.Sequential(
                nn.Conv2d(in_ch // 2, in_ch, 3, stride=(2,2),padding=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            )
            self.conv3x3 = _MultiScaleConv(in_ch * 3, in_ch // 2, 3)
            self.conv5x5 = _MultiScaleConv(in_ch * 3, in_ch // 2, 5)
            self.conv7x7 = _MultiScaleConv(in_ch * 3, in_ch // 2, 7)
            self.gap = _MaxPooling(in_ch * 3, in_ch // 2)
        else:
            self.conv3x3 = _MultiScaleConv(in_ch * 2, in_ch // 2, 3)
            self.conv5x5 = _MultiScaleConv(in_ch * 2, in_ch // 2, 5)
            self.conv7x7 = _MultiScaleConv(in_ch * 2, in_ch // 2, 7)
            self.gap = _MaxPooling(in_ch * 2, in_ch // 2)

        self.conv1x1_2 = _SingleConv(in_ch * 2, in_ch, 1)

        self.pv_block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch // 4, 3, padding=1, groups=in_ch // 4, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch)
        )
        self.art_block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch // 4, 3, padding=1, groups=in_ch // 4, bias=False),
            nn.BatchNorm2d(in_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pv_feature, art_feature, last_fusion=None):
        pv_residual = pv_feature
        art_residual = art_feature
        if last_fusion is not None:
            last_fusion = self.fusion_dowmsample(last_fusion)
            concat_1 = torch.cat([pv_feature, art_feature, last_fusion], dim=1)
        else:
            concat_1 = torch.cat([pv_feature, art_feature], dim=1)
        gap = self.gap(concat_1)
        c_3x3 = self.conv3x3(concat_1)
        c_5x5 = self.conv5x5(concat_1)
        c_7x7 = self.conv7x7(concat_1)
        concat_2 = torch.cat([c_3x3, c_5x5, c_7x7, gap], dim=1)
        fusion = self.conv1x1_2(concat_2)
        pv_b = self.pv_block(fusion)
        art_b = self.art_block(fusion)
        pv_b += pv_residual
        art_b += art_residual

        return self.relu(pv_b), self.relu(art_b), fusion


class Encoder(nn.Module):

    scale = 32

    def __init__(self):
        super().__init__()
        self.pv_base = models.resnet34(pretrained=True)
        self.pv_block_1 = nn.Sequential(*list(self.pv_base.children())[:3])
        self.pv_pooling = nn.Sequential(*list(self.pv_base.children())[3:4])
        self.pv_block_2 = nn.Sequential(*list(self.pv_base.children())[4:5])
        self.pv_block_3 = nn.Sequential(*list(self.pv_base.children())[5:6])
        self.pv_block_4 = nn.Sequential(*list(self.pv_base.children())[6:7])
        self.pv_block_5 = nn.Sequential(*list(self.pv_base.children())[7:8])

        self.art_base = models.resnet34(pretrained=True)
        self.art_block_1 = nn.Sequential(*list(self.art_base.children())[:3])
        self.art_pooling = nn.Sequential(*list(self.art_base.children())[3:4])
        self.art_block_2 = nn.Sequential(*list(self.art_base.children())[4:5])
        self.art_block_3 = nn.Sequential(*list(self.art_base.children())[5:6])
        self.art_block_4 = nn.Sequential(*list(self.art_base.children())[6:7])
        self.art_block_5 = nn.Sequential(*list(self.art_base.children())[7:8])

        # self.multi_scale_fusion_block_1 = Multi_Scale_Fusion_Block(self.scale * 2)
        self.multi_scale_fusion_block_2 = Multi_Scale_Fusion_Block(self.scale * 4)
        self.multi_scale_fusion_block_3 = Multi_Scale_Fusion_Block(self.scale * 8, True)
        self.multi_scale_fusion_block_4 = Multi_Scale_Fusion_Block(self.scale * 16, True)
        # self.dropout = nn.Dropout(0.2)
    
    def forward(self, pv_input, art_input):

        pv_c1 = self.pv_block_1(pv_input)
        art_c1 = self.art_block_1(art_input)
        pv_pool = self.pv_pooling(pv_c1)
        art_pool = self.art_pooling(art_c1)
        # pv_d1 = self.dropout(pv_pool)
        pv_c2 = self.pv_block_2(pv_pool)
        art_c2 = self.art_block_2(art_pool)
        # pv_c2, art_c2, fusion_1 = self.multi_scale_fusion_block_1(pv_c2, art_c2)
        # pv_d2 = self.dropout(pv_pool)
        pv_c3 = self.pv_block_3(pv_c2)
        art_c3 = self.art_block_3(art_c2)
        pv_c3, art_c3, fusion_2 = self.multi_scale_fusion_block_2(pv_c3, art_c3)
        # pv_d3 = self.dropout(pv_pool)
        pv_c4 = self.pv_block_4(pv_c3)
        art_c4 = self.art_block_4(art_c3)
        pv_c4, art_c4, fusion_3 = self.multi_scale_fusion_block_3(pv_c4, art_c4, fusion_2)
        # pv_d4 = self.dropout(pv_pool)
        pv_c5 = self.pv_block_5(pv_c4)
        art_c5 = self.art_block_5(art_c4)
        pv_c5, art_c5, fusion_4 = self.multi_scale_fusion_block_4(pv_c5, art_c5, fusion_3)
        # pv_predict = self.pv_classifier(pv_c5)
        # art_predict = self.art_classifier(art_c5)
        # fusion_predict = self.multi_scale_fusion_classifier(fusion_4)

        return  pv_c1, pv_c2, pv_c3, pv_c4, pv_c5, art_c1, art_c2, art_c3, art_c4, art_c5


class _Co_Learning_Block(nn.Module):
    def __init__(self, in_ch, out_ch, n_class):
        super().__init__()

        self.n_class = n_class
        self.pv_conv1_1 = _SingleConv(in_ch, out_ch, 1)
        self.pv_conv1_2 = _SingleConv(out_ch, out_ch // 2, 1)
        self.pv_conv1_3 = _SingleConv(out_ch * (n_class - 1), out_ch, 1)
        self.art_conv1_1 = _SingleConv(in_ch, out_ch, 1)
        self.art_conv1_2 = _SingleConv(out_ch, out_ch // 2, 1)
        self.art_conv1_3 = _SingleConv(out_ch * (n_class - 1), out_ch, 1)
        self.pv_classifier = nn.Conv2d(out_ch, n_class, 1)
        self.art_classifier = nn.Conv2d(out_ch, n_class, 1)



        self.pv_conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        self.art_conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, pv_merge, art_merge, pv_skip, art_skip, pv_mask, art_mask):

        pv_merge = self.pv_conv1_1(pv_merge)
        pv_feature = self.pv_conv3(pv_merge)
        pv_pred = self.pv_classifier(pv_feature)

        art_merge = self.art_conv1_1(art_merge)
        art_feature = self.art_conv3(art_merge)
        art_pred = self.art_classifier(art_feature)

        pv_skip = self.pv_conv1_2(pv_skip)
        art_skip = self.art_conv1_2(art_skip)

        # self.training = False
        pv_colearning, art_colearning, pv_similarity, art_similarity = None, None, None, None

        # binary prediction
        if self.training:
            mask_W = pv_pred.size()[3]
            for i in range(1, self.n_class):
                pv_mask_i = (pv_mask == i).float()
                art_mask_i = (art_mask == i).float()
                pv_mask_i = F.adaptive_max_pool2d(pv_mask_i, output_size=mask_W)
                art_mask_i = F.adaptive_max_pool2d(art_mask_i, output_size=mask_W)
                pv_colearning_i, art_colearning_i, pv_similarity_i, art_similarity_i = self.get_similarity_feature(pv_skip, art_skip, pv_mask_i, art_mask_i, pv_feature, art_feature)
                
                if pv_colearning is not None:
                    pv_colearning = torch.cat((pv_colearning, pv_colearning_i), dim=1)
                    art_colearning = torch.cat((art_colearning, art_colearning_i), dim=1)
                    pv_similarity = torch.cat((pv_similarity, pv_similarity_i), dim=1)
                    art_similarity = torch.cat((art_similarity, art_similarity_i), dim=1)
                else:
                    pv_colearning = pv_colearning_i
                    art_colearning = art_colearning_i
                    pv_similarity = pv_similarity_i
                    art_similarity = art_similarity_i
        else:
            pv_pred_temp = F.softmax(pv_pred, dim=1)
            art_pred_temp = F.softmax(art_pred, dim=1)
            pv_max = torch.max(pv_pred_temp, dim=1, keepdim=True)[1]
            art_max = torch.max(art_pred_temp, dim=1, keepdim=True)[1]
            for i in range(1, self.n_class):
                pv_mask_i = (pv_max == i).float()
                art_mask_i = (art_max == i).float()
                pv_colearning_i, art_colearning_i, pv_similarity_i, art_similarity_i = self.get_similarity_feature(pv_skip, art_skip, pv_mask_i, art_mask_i, pv_feature, art_feature)
                if pv_colearning is not None:
                    pv_colearning = torch.cat((pv_colearning, pv_colearning_i), dim=1)
                    art_colearning = torch.cat((art_colearning, art_colearning_i), dim=1)
                    pv_similarity = torch.cat((pv_similarity, pv_similarity_i), dim=1)
                    art_similarity = torch.cat((art_similarity, art_similarity_i), dim=1)
                else:
                    pv_colearning = pv_colearning_i
                    art_colearning = art_colearning_i
                    pv_similarity = pv_similarity_i
                    art_similarity = art_similarity_i
        pv_colearning = self.pv_conv1_3(pv_colearning)
        art_colearning = self.art_conv1_3(art_colearning)


        return pv_colearning, art_colearning, pv_pred, art_pred, pv_similarity, art_similarity

    def get_similarity_feature(self, pv_skip, art_skip, pv_mask, art_mask, pv_feature, art_feature):

        # masked average pooling
        pv_skip_masked = pv_skip * pv_mask
        art_skip_masked = art_skip * art_mask

        B, C, H, W = pv_skip_masked.size()

        pv_skip_masked = pv_skip_masked.view(B, C, -1)
        art_skip_masked = art_skip_masked.view(B, C, -1)

        pv_mask = pv_mask.view(B, -1)
        art_mask = art_mask.view(B, -1)

        pv_vector = torch.sum(pv_skip_masked, dim=2) / (1e-4 + torch.sum(pv_mask, dim=1)).unsqueeze(dim=1).expand([B, C])
        art_vector = torch.sum(art_skip_masked, dim=2) / (1e-4 + torch.sum(art_mask, dim=1)).unsqueeze(dim=1).expand([B, C])

        # calculate cosine similarity
        pv_vector = pv_vector.unsqueeze(dim=2).unsqueeze(dim=3).expand([B, C, H, W])
        art_vector = art_vector.unsqueeze(dim=2).unsqueeze(dim=3).expand([B, C, H, W])

        pv_similarity = F.cosine_similarity(pv_skip, art_vector, dim=1).unsqueeze(dim=1)
        art_similarity = F.cosine_similarity(art_skip, pv_vector, dim=1).unsqueeze(dim=1)

        pv_feature = (pv_feature * pv_similarity) + pv_feature
        art_feature = (art_feature * art_similarity) + art_feature

        return pv_feature, art_feature, pv_similarity, art_similarity


# multimodal fusion unet
class Cross_Modal_Guidance_CHAOS(nn.Module):

    scale = 32

    def __init__(self, in_ch, out_ch, **args):
        super().__init__()

        # siamese-encoder
        self.encoder = Encoder()

        # pv-decoder
        self.pv_conv6 = _DoubleConv(self.scale * 16 + self.scale * 8, self.scale * 8)
        self.pv_conv7 = _DoubleConv(self.scale * 8 * 2 + self.scale * 4, self.scale * 4)
        self.pv_conv8 = _DoubleConv(self.scale * 4 * 2 + self.scale * 2, self.scale * 2)
        self.pv_conv9 = _DoubleConv(self.scale * 2 + self.scale * 2, self.scale * 2)
        self.pv_conv10 = nn.Conv2d(self.scale * 2, out_ch, 1, bias=False)

        # art-decoder
        self.art_conv6 = _DoubleConv(self.scale * 16 + self.scale * 8, self.scale * 8)
        self.art_conv7 = _DoubleConv(self.scale * 8 * 2 + self.scale * 4, self.scale * 4)
        self.art_conv8 = _DoubleConv(self.scale * 4 * 2 + self.scale * 2, self.scale * 2)
        self.art_conv9 = _DoubleConv(self.scale * 2 + self.scale * 2, self.scale * 2)
        self.art_conv10 = nn.Conv2d(self.scale * 2, out_ch, 1, bias=False)

        # fusion-block
        self.co_conv1 = _Co_Learning_Block(self.scale * 16 + self.scale * 8, self.scale * 8, out_ch)
        self.co_conv2 = _Co_Learning_Block(self.scale * 8 * 2 + self.scale * 4, self.scale * 4, out_ch)
        # self.co_conv3 = _Co_Learning_Block(self.scale * 4 * 2 + self.scale * 2, self.scale * 2)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, pv_imgs, art_imgs, pv_mask, art_mask):

        # B, C, H, W = pv_imgs.size()
        # pv_imgs = pv_imgs.expand(B, 3, H, W)
        # art_imgs = art_imgs.expand(B, 3, H, W)

        # encoder
        pv_c1, pv_c2, pv_c3, pv_c4, pv_c5, art_c1, art_c2, art_c3, art_c4, art_c5 = self.encoder(pv_imgs, art_imgs)

        #########################################################################################

        # co-learning decoder
        pv_up_6 = self.up2(pv_c5)
        pv_merge6 = torch.cat([pv_up_6, pv_c4], dim=1)
        art_up_6 = self.up2(art_c5)
        art_merge6 = torch.cat([art_up_6, art_c4], dim=1)
        # pv_merge6 = self.drop(pv_merge6)
        # art_merge6 = self.drop(art_merge6)
        pv_c6 = self.pv_conv6(pv_merge6)
        art_c6 = self.art_conv6(art_merge6)
        pv_co1, art_co1, pv_pred_2, art_pred_2, pv_simi_1, art_simi_1 = self.co_conv1(pv_merge6, art_merge6, pv_c4, art_c4, pv_mask, art_mask)
        pv_c6 = torch.cat([pv_c6, pv_co1], dim=1)
        art_c6 = torch.cat([art_c6, art_co1], dim=1)

        pv_up_7 = self.up2(pv_c6)
        pv_merge7 = torch.cat([pv_up_7, pv_c3], dim=1)
        art_up_7 = self.up2(art_c6)
        art_merge7 = torch.cat([art_up_7, art_c3], dim=1)
        # pv_merge7 = self.drop(pv_merge7)
        # art_merge7 = self.drop(art_merge7)
        pv_c7 = self.pv_conv7(pv_merge7)
        art_c7 = self.art_conv7(art_merge7)
        pv_co2, art_co2, pv_pred_3, art_pred_3, pv_simi_2, art_simi_2 = self.co_conv2(pv_merge7, art_merge7, pv_c3, art_c3, pv_mask, art_mask)
        pv_c7 = torch.cat([pv_c7, pv_co2], dim=1)
        art_c7 = torch.cat([art_c7, art_co2], dim=1)

        pv_up_8 = self.up2(pv_c7)
        pv_merge8 = torch.cat([pv_up_8, pv_c2], dim=1)
        art_up_8 = self.up2(art_c7)
        art_merge8 = torch.cat([art_up_8, art_c2], dim=1)
        # pv_merge8 = self.drop(pv_merge8)
        # art_merge8 = self.drop(art_merge8)
        pv_c8 = self.pv_conv8(pv_merge8)
        art_c8 = self.art_conv8(art_merge8)
        # pv_co3, art_co3, pv_pred_3, art_pred_3, pv_simi_3, art_simi_3 = self.co_conv3(pv_merge8, art_merge8, pv_c2, art_c2)
        # pv_c8 = torch.cat([pv_c8, pv_co3], dim=1)
        # art_c8 = torch.cat([art_c8, art_co3], dim=1)

        pv_up_9 = self.up2(pv_c8)
        pv_merge9 = torch.cat([pv_up_9, pv_c1], dim=1)
        pv_c9 = self.pv_conv9(pv_merge9)
        pv_c10 = self.pv_conv10(pv_c9)
        art_up_9 = self.up2(art_c8)
        art_merge9 = torch.cat([art_up_9, art_c1], dim=1)
        art_c9 = self.art_conv9(art_merge9)
        art_c10 = self.art_conv10(art_c9)


        pred_2 = (pv_pred_2, art_pred_2)
        pred_3 = (pv_pred_3, art_pred_3)
        pv_final_seg = F.interpolate(pv_c10, pv_mask.size()[2:], mode='bilinear', align_corners=True)
        art_final_seg = F.interpolate(art_c10, art_mask.size()[2:], mode='bilinear', align_corners=True)
        pred_4 = (pv_final_seg, art_final_seg)

        # out = (pv_simi_1, art_simi_1, pv_simi_2, art_simi_2)
        
        out = (pred_2, pred_3, pred_4)
        out_mask = (pv_mask, art_mask)
        metric_imgs = F.interpolate(pv_c10, pv_mask.size()[2:], mode='bilinear', align_corners=True)
        # metric_imgs = torch.sigmoid(metric_imgs)


        # predicts, targets, metric_imgs, metric_targets
        return out, out_mask, metric_imgs, pv_mask



if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    a = torch.ones((2, 1, 512, 512)).to(device)
    b = torch.ones((2, 1, 512, 512)).to(device)
    c = torch.ones((2, 1, 512, 512)).to(device)
    d = torch.ones((2, 1, 512, 512)).to(device)
    model = Cross_Modal_Guidance_CHAOS(1, 5).to(device)
    # out, out_mask, c10, pv_mask = model(a, b, c, d)
    out, out_mask, pv_c10, pv_mask =  model(a, b, c, d)
    
    pred_1, pred_2, pred_3 = out
    pv_pred_1, art_pred_1 = pred_1
    pv_pred_2, art_pred_2 = pred_2
    pv_pred_3, art_pred_3 = pred_3
    pv_mask, art_mask = out_mask
    print(pv_pred_1.size())
    print(pv_pred_2.size())
    print(pv_pred_3.size())
    print(pv_mask.size())
    print(pv_c10.size())
    print(pv_mask.size())