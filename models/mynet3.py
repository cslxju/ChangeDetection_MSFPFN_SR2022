import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

class F_mynet3(nn.Module):
    def __init__(self, backbone='resnet18', in_c=3, f_c=64, output_stride=8):
        self.in_c = in_c
        super(F_mynet3, self).__init__()
        self.module = mynet3(backbone=backbone, output_stride=output_stride, f_c=f_c, in_c=self.in_c)

    def forward(self, inputA, inputB):
        return self.module(inputA, inputB)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def ResNet34(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 64
    """
    print(in_c)
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model


def ResNet18(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet18'])
    return model


def ResNet50(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3):
    """
    output, low_level_feat:
    2048, 256
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c)
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, in_c=3):

        self.inplanes = 64
        self.in_c = in_c
        print('in_c: ', self.in_c)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 4:
            strides = [1, 1, 1, 1]
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError
        size = [64, 128, 256, 512]
        # Modules

        self.conv1 = conv_block_nested(self.in_c, self.inplanes, self.inplanes)

        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.conv64_3_64 = conv_block_nested(size[0] * 3, size[0], size[0])
        self.conv128_3_128 = conv_block_nested(size[1] * 3, size[1], size[1])
        self.conv256_3_512 = conv_block_nested(size[2] * 3, size[2], size[2])
        self.conv512_3_512 = conv_block_nested(size[3] * 3, size[3], size[3])


        self.conv64_96=conv_block_nested(64,96,96)
        self.conv128_96=conv_block_nested(128,96,96)
        self.conv256_96=conv_block_nested(256,96,96)
        self.conv512_96=conv_block_nested(512,96,96)
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, inputA, inputB):
        '''A'''
        xA = self.conv1(inputA)  # [64,128,128]
        xA = self.bn1(xA)
        xA = self.relu(xA)
        xA = self.maxpool(xA)  # | 4   [64,64,64]
        xA1 = self.layer1(xA)  # | 4        [64,64,64]
        xA2 = self.layer2(xA1)  # | 8      [128,32,32]

        xA3 = self.layer3(xA2)  # | 16    [256,16,16]

        xA4 = self.layer4(xA3)  # | 32      [512,8,8]
        '''B'''
        xB = self.conv1(inputB)  # [64,128,128]
        xB = self.bn1(xB)
        xB = self.relu(xB)

        xB = self.maxpool(xB)  # | 4   [64,64,64]
        xB1 = self.layer1(xB)  # | 4        [64,64,64]

        xB2 = self.layer2(xB1)  # | 8      [128,32,32]

        xB3 = self.layer3(xB2)  # | 16    [256,16,16]

        xB4 = self.layer4(xB3)  # | 32      [512,8,8]
       

        low_level_feat1 = self.conv64_96(self.conv64_3_64(torch.cat((torch.abs(xA1 - xB1), xA1, xB1), 1)))
        low_level_feat2 = self.conv128_96(self.conv128_3_128(torch.cat((torch.abs(xA2 - xB2), xA2, xB2), 1)))
        low_level_feat3 = self.conv256_96(self.conv256_3_512(torch.cat((torch.abs(xA3 - xB3),  xA3, xB3), 1))) 
        low_level_feat4 = self.conv512_96(self.conv512_3_512(torch.cat((torch.abs(xA4 - xB4), xA4, xB4), 1)))

        return low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def build_backbone(backbone, output_stride, BatchNorm, in_c=3):
    if backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm, in_c=in_c)
    elif backbone == 'resnet34':
        return ResNet34(output_stride, BatchNorm, in_c=in_c)
    elif backbone == 'resnet18':
        return ResNet18(output_stride, BatchNorm, in_c=in_c)
    else:
        raise NotImplementedError


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(out)

class residual_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(residual_block, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)

        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x1 = self.conv10(x)
        x1 = self.bn2(x1)
        x2 = self.conv3(x1)
        identity = x1
        x2 = self.bn2(x2)
        x3 = self.conv1(x2)
        x3 = self.bn2(x3)
        x3 = self.activation(x1+x3)


        x4 = self.conv3(x3)
        x4 = self.bn2(x4)

        x5 = self.conv1(x4)
        x5 = self.bn2(x5)


        output = self.activation(x5 + identity)
        return output


class Decoder(nn.Module):
    def __init__(self, fc, BatchNorm):
        super(Decoder, self).__init__()
        self.fc = fc
        self.cam = ChannelAttention(384,16)
        self.in_c = 384
        self.out_c = 96
        self.last_conv1 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                        BatchNorm(1),
                                        nn.ReLU(),
                                        )
        self.last_conv4 = nn.Sequential(nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(48),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0, bias=False),
                                        BatchNorm(1),
                                        nn.ReLU(),
                                        )
        self.sa = SpatialAttention()
    
        self.pool1 = nn.AvgPool2d(1,1)
        self.pool2 = nn.AvgPool2d(2,2)
        self.pool4 = nn.AvgPool2d(4,4)
        self.pool8 = nn.AvgPool2d(8,8)
        self.convert = nn.Conv2d(96, 96, 1)
        self.con1 = conv_block_nested(self.in_c,self.in_c,self.in_c)

        self.con1x1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, kernel_size=1,padding=0, bias=True),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU()
        )
        self.con3x3 = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, kernel_size=3,padding=1, bias=True),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU()
        )
        self.con5x5 = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, kernel_size=5,padding=2, bias=True),
            nn.BatchNorm2d(self.out_c),
            nn.ReLU()
        )
        self.con11 = nn.Conv2d(self.out_c,self.out_c,1)
        self.bn11 = nn.BatchNorm2d(self.out_c)
        self.relu11 = nn.ReLU()
        self.residual_block = residual_block(self.in_c, self.out_c)
   



        self._init_weight()

    def forward(self, x1, x2, x3, x4):
  
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.cam(x)

        x_out1 = self.last_conv1(x)

        x_0 = self.residual_block(x)
        x_1 = self.pool1(x)
        x_2 = self.pool2(x)
        x_4 = self.pool4(x)
        x_8 = self.pool8(x)

        x_1 = self.con5x5(x_1)
        x_2 = self.con3x3(x_2)
        x_4 = self.con3x3(x_4)
        x_8 = self.con1x1(x_8)

        x_8 = F.interpolate(x_8, size=x_4.size()[2:], mode='bilinear', align_corners=True)
        z1 = self.bn11(self.con11(x_4)+x_8)
        z1 = F.interpolate(z1, size=x_2.size()[2:], mode='bilinear', align_corners=True)
        z2 = self.bn11(self.con11(x_2)+z1)
        z2 = F.interpolate(z2, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        z3 = self.bn11(self.con11(x_1)+z2)
        z3 = F.interpolate(z3, size=x_0.size()[2:], mode='bilinear', align_corners=True)
        z4 = self.relu11(self.bn11(self.con11(x_0)+z3))
        x_out2 = self.last_conv4(z4)

        return x_out1,x_out2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(fc, backbone, BatchNorm):
    return Decoder(fc, BatchNorm)


class mynet3(nn.Module):
    def __init__(self, backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(mynet3, self).__init__()
        print('arch: mynet3')
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c)
        self.decoder = build_decoder(f_c, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputA, inputB):
        x, f2, f3, f4 = self.backbone(inputA, inputB)
        x1,x2 = self.decoder(x, f2, f3, f4)
        return x1,x2

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


