import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from utils import trunc_normal_


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads.cuda()], dim=1))

    return out


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN1':
        out = nn.BatchNorm1d(inplanes)
    elif norm_cfg == 'BN2':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'BN3':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN' or norm_cfg == 'IN3':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out

def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out



class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        # weight = self.weight
        # weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        # weight = weight - weight_mean
        # # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        # std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        # w = weight / std.expand_as(weight)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=(1, 1, 1), padding=1, bias=False,
                               weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.conv3 = conv3x3x3(planes, planes * 4, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm3 = Norm_layer(norm_cfg, planes * 4)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.nonlin(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out

class res3d_tea(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 shortcut_type='B',
                 norm_cfg='BN3',
                 activation_cfg='ReLU',
                 img_size=None,
                 weight_std=False):
        super(res3d_tea, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(1, 1, 1), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU', weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std), Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=True, logger=logger)
            pass
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                    m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.gap(x).flatten(1)
        return out

class res3d(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=1,
                 shortcut_type='B',
                 norm_cfg='BN3',
                 activation_cfg='ReLU',
                 img_size=None,
                 weight_std=False):
        super(res3d, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False,
                               weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(1, 1, 1), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=(2, 2, 2), norm_cfg=norm_cfg,
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU', weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std), Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=True, logger=logger)
            pass
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                    m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        out_1 = self.gap(x).flatten(1)
        x = self.layer2(x)
        out_2 = self.gap(x).flatten(1)
        x = self.layer3(x)
        out_3 = self.gap(x).flatten(1)
        x = self.layer4(x)
        out_4 = self.gap(x).flatten(1)
        return [out_1, out_2, out_3, out_4]



def res3d10(**kwargs):
    model = res3d(depth=10, in_channels=1, shortcut_type='B', norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False, **kwargs)
    return model

def res3d18(**kwargs):
    model = res3d(depth=18, in_channels=1, shortcut_type='B', norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False, **kwargs)
    return model

def res3d34(**kwargs):
    model = res3d(depth=34, in_channels=1, shortcut_type='B', norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False, **kwargs)
    return model

def res3d50(**kwargs):
    model = res3d(depth=50, in_channels=1, shortcut_type='B', norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False, **kwargs)
    return model
def res3d50_tea(**kwargs):
    model = res3d_tea(depth=50, in_channels=1, shortcut_type='B', norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False, **kwargs)
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



class PIXELHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Conv3d(in_dim, out_dim, kernel_size=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.mlp(x)
        return x

if __name__ == '__main__':
    x = torch.zeros(2,1,16,96,96)
    model = res3d50()
    model(x)