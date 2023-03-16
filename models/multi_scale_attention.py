import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnext101_32x8d

class _EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class semanticModule(nn.Module):
    """
    Semantic attention module
    """
    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim

        self.enc1 = _EncoderBlock(in_dim, in_dim*2)
        self.enc2 = _EncoderBlock(in_dim*2, in_dim*4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim )

    def forward(self,x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2( enc2)
        dec1 = self.dec1( F.interpolate(dec2, enc1.size()[2:], mode='bilinear', align_corners=True))

        return enc2.view(-1), dec1

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
       
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention
    
    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, use_pam = True):
        super(PAM_CAM_Layer, self).__init__()
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.attn(x)
    
class MultiConv(nn.Module):
    """
    Helper function for Multiple Convolutions for refining.
    
    Parameters:
    ----------
    inputs:
        in_ch : input channels
        out_ch : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """
    def __init__(self, in_ch, out_ch, attn = True):
        super(MultiConv, self).__init__()
        
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1), 
            nn.BatchNorm2d(64), 
            nn.Softmax2d() if attn else nn.PReLU()
        )
    
    def forward(self, x):
        return self.fuse_attn(x)


def resnext101():
    model  = resnext101_32x8d()
    model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias = False)
    model.avgpool = nn.AvgPool2d((7,7), (1, 1))
    model.fc = nn.Sequential(
        Lambda(lambda  x: x.view(x.size(0), -1)),
        Lambda(lambda  x: x.view(1, -1) if 1 == len(x.size()) else x),
        nn.Linear(2048, 1000)
    )
    return model


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))

class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = resnext101()
        
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4



class Multi_Scale_Attention_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=5):
        super().__init__()
        self.resnext = ResNeXt101()

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        inter_channels = 64
        out_channels=64

        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv8_1=nn.Conv2d(64,64,1)
        self.conv8_2=nn.Conv2d(64,64,1)
        self.conv8_3=nn.Conv2d(64,64,1)
        self.conv8_4=nn.Conv2d(64,64,1)
        self.conv8_11=nn.Conv2d(64,64,1)
        self.conv8_12=nn.Conv2d(64,64,1)
        self.conv8_13=nn.Conv2d(64,64,1)
        self.conv8_14=nn.Conv2d(64,64,1)

        self.softmax_1 = nn.Softmax(dim=-1)

        self.pam_attention_1_1= PAM_CAM_Layer(64, True)
        self.cam_attention_1_1= PAM_CAM_Layer(64, False)
        self.semanticModule_1_1 = semanticModule(128)
        
        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
    
    #Dual Attention mechanism
        self.pam_attention_1_2 = PAM_CAM_Layer(64)
        self.cam_attention_1_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_3 = PAM_CAM_Layer(64)
        self.cam_attention_1_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_4 = PAM_CAM_Layer(64)
        self.cam_attention_1_4 = PAM_CAM_Layer(64, False)
        
        self.pam_attention_2_1 = PAM_CAM_Layer(64)
        self.cam_attention_2_1 = PAM_CAM_Layer(64, False)
        self.semanticModule_2_1 = semanticModule(128)
        
        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.pam_attention_2_2 = PAM_CAM_Layer(64)
        self.cam_attention_2_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_3 = PAM_CAM_Layer(64)
        self.cam_attention_2_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_4 = PAM_CAM_Layer(64)
        self.cam_attention_2_4 = PAM_CAM_Layer(64, False)
        
        self.fuse1 = MultiConv(256, 64, False)

        self.attention4 = MultiConv(128, 64)
        self.attention3 = MultiConv(128, 64)
        self.attention2 = MultiConv(128, 64)
        self.attention1 = MultiConv(128, 64)

        self.refine4 = MultiConv(128, 64, False)
        self.refine3 = MultiConv(128, 64, False)
        self.refine2 = MultiConv(128, 64, False)
        self.refine1 = MultiConv(128, 64, False)

        self.predict4 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 5, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, t1_img, t2_img, t1_mask, t2_mask):
        
        # input = torch.cat([t1_img.float(), t2_img.float()], dim=1)

        layer0 = self.resnext.layer0(t1_img)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.interpolate(self.down4(layer4), size=layer1.size()[2:], mode='bilinear', align_corners=True)
        down3 = F.interpolate(self.down3(layer3), size=layer1.size()[2:], mode='bilinear', align_corners=True)
        down2 = F.interpolate(self.down2(layer2), size=layer1.size()[2:], mode='bilinear', align_corners=True)
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        semVector_1_1,semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1),1))


        attn_pam4 = self.pam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_cam4 = self.cam_attention_1_4(torch.cat((down4, fuse1), 1))

        attention1_4=self.conv8_1((attn_cam4+attn_pam4)*self.conv_sem_1_1(semanticModule_1_1))

        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_pam3 = self.pam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_cam3 = self.cam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3=self.conv8_2((attn_cam3+attn_pam3)*self.conv_sem_1_2(semanticModule_1_2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        attn_pam2 = self.pam_attention_1_2(torch.cat((down2, fuse1), 1))
        attn_cam2 = self.cam_attention_1_2(torch.cat((down2, fuse1), 1))
        attention1_2=self.conv8_3((attn_cam2+attn_pam2)*self.conv_sem_1_3(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        attn_pam1 = self.pam_attention_1_1(torch.cat((down1, fuse1), 1))
        attn_cam1 = self.cam_attention_1_1(torch.cat((down1, fuse1), 1))
        attention1_1 = self.conv8_4((attn_cam1+attn_pam1) * self.conv_sem_1_4(semanticModule_1_4))
        
        ##new design with stacked attention

        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4 * fuse1), 1))

        refine4_1 = self.pam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4_2 = self.cam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4 = self.conv8_11((refine4_1+refine4_2) * self.conv_sem_2_1(semanticModule_2_1))

        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.pam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3_2 = self.cam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3 = self.conv8_12((refine3_1+refine3_2) * self.conv_sem_2_2(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_1 = self.pam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2_2 = self.cam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2 = self.conv8_13((refine2_1+refine2_2)*self.conv_sem_2_3(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_1 = self.pam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        refine1_2 = self.cam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))

        refine1=self.conv8_14((refine1_1+refine1_2) * self.conv_sem_2_4(semanticModule_2_4))
        
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        predict1 = F.interpolate(predict1, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=t1_img.size()[2:], mode='bilinear', align_corners=True)

        predict1_2 = F.interpolate(predict1_2, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        predict2_2 = F.interpolate(predict2_2, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        predict3_2 = F.interpolate(predict3_2, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        predict4_2 = F.interpolate(predict4_2, size=t1_img.size()[2:], mode='bilinear', align_corners=True)
        

        output = semVector_1_1,\
                   semVector_2_1, \
                   semVector_1_2, \
                   semVector_2_2, \
                   semVector_1_3, \
                   semVector_2_3, \
                   semVector_1_4, \
                   semVector_2_4, \
                   torch.cat((down1, fuse1), 1),\
                   torch.cat((down2, fuse1), 1),\
                   torch.cat((down3, fuse1), 1),\
                   torch.cat((down4, fuse1), 1), \
                   torch.cat((down1, attention1_1 * fuse1), 1), \
                   torch.cat((down2, attention1_2 * fuse1), 1), \
                   torch.cat((down3, attention1_3 * fuse1), 1), \
                   torch.cat((down4, attention1_4 * fuse1), 1), \
                   semanticModule_1_4, \
                   semanticModule_1_3, \
                   semanticModule_1_2, \
                   semanticModule_1_1, \
                   semanticModule_2_4, \
                   semanticModule_2_3, \
                   semanticModule_2_2, \
                   semanticModule_2_1, \
                   predict1, \
                   predict2, \
                   predict3, \
                   predict4, \
                   predict1_2, \
                   predict2_2, \
                   predict3_2, \
                   predict4_2
        
        metric_imgs = (predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4
        
        # output, targets, metric_imgs, metric_mask
        return output, t1_mask, metric_imgs, t1_mask

if __name__ == '__main__':

    
    device = torch.device("cpu")
    a = torch.ones((2, 1, 256, 256)).to(device)
    b = torch.ones((2, 1, 256, 256)).to(device)
    c = torch.ones((2, 1, 256, 256)).to(device)
    d = torch.ones((2, 1, 256, 256)).to(device)
    model = Multi_Scale_Attention_Net(3, 5).to(device)
    # out, out_mask, c10, pv_mask = model(a, b, c, d)
    output, t1_mask, metric_imgs, t1_mask =  model(a, b, c, d)
    
    # pv_mask, art_mask = out_mask
    print(metric_imgs.size())
    print(t1_mask.size())

    