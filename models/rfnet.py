import torch
import torch.nn as nn
import os

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=5):
        super(prm_generator_laststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*2, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(self.embedding_layer(y))
        return seg

class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=5):
        super(prm_generator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*2, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))
        return seg

####modal fusion in each region
class modal_fusion(nn.Module):
    def __init__(self, in_channel=64):
        super(modal_fusion, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(2*in_channel+1, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, 2, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name):
        B, K, C, H, W, Z = x.size()

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        ###we find directly using weighted sum still achieve competing performance
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat

###fuse region feature
class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=5):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)

class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=5):
        super(region_aware_modal_fusion, self).__init__()
        self.num_cls = num_cls

        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel) for i in range(num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
                        general_conv3d(in_channel*2, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

        # self.clsname_list = ['BG', 'NCR/NET', 'ED', 'ET', "AA"] ##BRATS2020 and BRATS2018
        self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET'] ##BRATS2015

    def forward(self, x, prm, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        ###divide modal features into different regions
        t1 = y[:, 0:1, ...] * prm
        t2 = y[:, 1:2, ...] * prm

        modal_feat = torch.stack((t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        ###modal fusion in each region
        region_fused_feat = []
        for i in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i+1, ...], self.clsname_list[i]))
        region_fused_feat = torch.stack(region_fused_feat, dim=1)
        '''
        region_fused_feat = torch.stack((self.modal_fusion[0](region_feat[0], prm[:, 0:1, ...], 'BG'),
                                         self.modal_fusion[1](region_feat[1], prm[:, 1:2, ...], 'NCR/NET'),
                                         self.modal_fusion[2](region_feat[2], prm[:, 2:3, ...], 'ED'),
                                         self.modal_fusion[3](region_feat[3], prm[:, 3:4, ...], 'ET')), dim=1)
        '''

        ###gain final feat with a short cut
        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        return final_feat

basic_dims = 16
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=5):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=5):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4))

class RFNet(nn.Module):
    def __init__(self, in_ch, num_cls=5):
        super().__init__()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, t1_img, t2_img, t1_mask, t2_mask):
        
        #extract feature from different layers
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(t1_img)
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(t2_img)

        x1 = torch.stack((t1_x1, t2_x1), dim=1) #Bx4xCxHWZ
        x2 = torch.stack((t1_x2, t2_x2), dim=1)
        x3 = torch.stack((t1_x3, t2_x3), dim=1)
        x4 = torch.stack((t1_x4, t2_x4), dim=1)
        # mask = torch.stack((t1_mask, t2_mask), dim=1)
        
        mask = torch.tensor([[True, True]]).expand(t1_img.size(0), 2)
        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
        t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)

        output = fuse_pred, (t1_pred, t2_pred), prm_preds
        # output, targets, metric_imgs, metric_mask
        return output, t1_mask, fuse_pred, t1_mask
        
        # return fuse_pred


if __name__ == '__main__':

    # CUDA_LAUNCH_BLOCKING=1
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    a = torch.randn((2,1,24,24,24)).to(device)
    b = torch.randn((2,1,24,24,24)).to(device)
    c = torch.ones((1,5,24,24,24)).long().to(device)
    d = torch.ones((1,5,24,24,24)).long().to(device)
    mask = torch.tensor([[True, True]])

    model= RFNet(2, 5).to(device)
    output, targets, metric_imgs, metric_mask = model(a,b,c,d)
    fuse_pred, (t1_pred, t2_pred), prm_preds = output
    print(fuse_pred.size())
    #fuse_pred.size():torch.Size([1, 5, 24, 24, 24]) target-onehot

    # RFNet()