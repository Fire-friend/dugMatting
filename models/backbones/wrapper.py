import os
from functools import reduce

import torch
import torch.nn as nn
import torchvision

from .hrnetConfig import update_config
from .mobilenetv2 import MobileNetV2
from .mobilenetv3_ import MobileNetV3_Large
from .seg_hrnet import HighResolutionNet
from .swin_transformer import SwinTransformer


class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone_human(BaseBackbone):
    """ MobileNetV2 Backbone 
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone_human, self).__init__(in_channels)

        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]
        self.out_channels = [16, 24, 32, 96, 1280]
        self.out_channels_sum = sum(self.enc_channels)
        self.output_size_num = 5
        self.load_pretrained_ckpt()

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self, device=None):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/mobilenetv2_human_seg.ckpt'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()

        ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
            else:
                print(name)
        self.model.load_state_dict(new_params)


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV2Backbone, self).__init__(in_channels)
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        # self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]
        self.out_channels = [16, 24, 32, 96, 1280]
        self.out_channels_sum = sum(self.enc_channels)
        self.output_size_num = 5

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self, device=None):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = '/data/wjw/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()

        # if device is not None:
        #     map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
        #     ckpt = torch.load(ckpt_path, map_location=map_location)
        # else:
        ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
            else:
                print(name)
        self.model.load_state_dict(new_params)


class MobileNetV3LargeBackbone(BaseBackbone):
    """ MobileNetV3 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV3LargeBackbone, self).__init__(in_channels)

        # self.model = MobileNetV3_Large()
        self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.out_channels = [16, 24, 40, 112, 960]  # , 160
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[13](x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]


class MobileNetV3SmallBackbone(BaseBackbone):
    """ MobileNetV3 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV3SmallBackbone, self).__init__(in_channels)

        # self.model = MobileNetV3_Large()
        self.model = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.out_channels = [16, 16, 24, 48, 576]  # , 160
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        enc2x = x
        x = self.model.features[1](x)
        enc4x = x
        x = self.model.features[2](x)
        x = self.model.features[3](x)

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        enc8x = x
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        enc16x = x

        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        enc32x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        return [enc2x, enc4x, enc8x, enc16x, enc32x]


class MobileNetV3Backbone(BaseBackbone):
    """ MobileNetV3 Backbone
    """

    def __init__(self, in_channels):
        super(MobileNetV3Backbone, self).__init__(in_channels)

        self.model = MobileNetV3_Large()
        self.out_channels = [16, 24, 40, 160, 960]  # , 160
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.conv1(x)
        x = self.model.hs1(x)
        x = self.model.bn1(x)
        x = self.model.bneck[0](x)
        enc2x = x

        x = self.model.bneck[1](x)
        enc4x = x

        x = self.model.bneck[2](x)  #
        x = self.model.bneck[3](x)
        x = self.model.bneck[4](x)
        x = self.model.bneck[5](x)
        enc8x = x

        x = self.model.bneck[6](x)
        x = self.model.bneck[7](x)
        x = self.model.bneck[8](x)
        x = self.model.bneck[9](x)
        x = self.model.bneck[10](x)
        x = self.model.bneck[11](x)
        x = self.model.bneck[12](x)
        enc16x = x

        x = self.model.bneck[13](x)
        x = self.model.hs2(self.model.bn2(self.model.conv2(self.model.bneck[14](x))))
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/mbv3_large.old.pth.tar'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)['state_dict']
        else:
            ckpt = torch.load(ckpt_path)['state_dict']
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class ResnetBackbone(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(ResnetBackbone, self).__init__(in_channels)

        self.model = torchvision.models.resnet50(pretrained=True)
        self.out_channels = [64, 256, 512, 1024, 2048]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        # x = self.model.conv1(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        enc2x = x

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        enc4x = x

        x = self.model.layer2(x)
        enc8x = x

        x = self.model.layer3(x)
        enc16x = x

        x = self.model.layer4(x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/resnet50-0676ba61.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class ResnetBackbone_r34(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(ResnetBackbone_r34, self).__init__(in_channels)

        # self.model = torchvision.models.resnet50(pretrained=False)
        self.model = torchvision.models.resnet34(pretrained=True)
        self.out_channels = [64, 64, 128, 256, 512]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        # x = self.model.conv1(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        enc2x = x

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        enc4x = x

        x = self.model.layer2(x)
        enc8x = x

        x = self.model.layer3(x)
        enc16x = x

        x = self.model.layer4(x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/resnet34-b627a593.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class SwinT(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(SwinT, self).__init__(in_channels)

        # self.model = torchvision.models.resnet50(pretrained=False)
        self.model = SwinTransformer()
        self.out_channels = [96, 192, 384, 768]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

    def forward(self, x):
        enc2x, enc4x, enc8x, enc16x, = self.model(x)

        return [enc2x, enc4x, enc8x, enc16x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/resnet50-0676ba61.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class Convnext(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(Convnext, self).__init__(in_channels)

        # self.model = torchvision.models.resnet50(pretrained=False)
        self.model = torchvision.models.convnext_base(pretrained=True)
        self.out_channels = [64, 64, 128, 256, 512]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)

    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        # x = self.model.conv1(x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)

        x = self.model.features[2](x)
        x = self.model.features[3](x)

        x = self.model.features[4](x)
        x = self.model.features[5](x)

        x = self.model.features[6](x)
        x = self.model.features[7](x)

        x = self.model.bn1(x)
        x = self.model.relu(x)
        enc2x = x

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        enc4x = x

        x = self.model.layer2(x)
        enc8x = x

        x = self.model.layer3(x)
        enc16x = x

        x = self.model.layer4(x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/resnet34-b627a593.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class Efficient_b7(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(Efficient_b7, self).__init__(in_channels)

        # self.model = torchvision.models.resnet50(pretrained=false)
        self.model = torchvision.models.efficientnet_b7(pretrained=True)

        # self.model2 = EfficientNet.from_pretrained('efficientnet-b7')

        self.out_channels = [32, 48, 80, 224, 640]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=false)

    def forward(self, x):

        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x
        x = self.model.features[2](x)
        enc4x = x
        x = self.model.features[3](x)
        enc8x = x
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        enc16x = x
        x = self.model.features[6](x)
        x = self.model.features[7](x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/resnet34-b627a593.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class Efficient_b6(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(Efficient_b6, self).__init__(in_channels)

        # self.model = torchvision.models.resnet50(pretrained=false)
        self.model = torchvision.models.efficientnet_b6(pretrained=True)

        # self.model2 = EfficientNet.from_pretrained('efficientnet-b7')

        self.out_channels = [32, 40, 72, 200, 576]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=false)

    def forward(self, x):

        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x
        x = self.model.features[2](x)
        enc4x = x
        x = self.model.features[3](x)
        enc8x = x
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        enc16x = x
        x = self.model.features[6](x)
        x = self.model.features[7](x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]  # , enc32x

    def load_pretrained_ckpt(self, device):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/resnet34-b627a593.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path)
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                print('module.' + name)
        self.model.load_state_dict(new_params)


class HRnet18(BaseBackbone):
    """ MobileNetV2 Backbone
    """

    def __init__(self, in_channels):
        super(HRnet18, self).__init__(in_channels)
        from models.backbones.hrnetConfig import config
        # self.model = torchvision.models.resnet50(pretrained=false)
        update_config(config, './config/seg_hrnet_w18.yaml')
        self.model = HighResolutionNet(config, in_channels)

        # self.model2 = EfficientNet.from_pretrained('efficientnet-b7')

        self.out_channels = [18, 36, 72, 144]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=false)

        self.load_pretrained_ckpt()

    def forward(self, x):
        x_list = self.model(x)
        return x_list

    def load_pretrained_ckpt(self, device=None):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/HRNet_W18_C_ssld_pretrained.pth'
        assert os.path.exists(ckpt_path), 'cannot find the pretrained backbone from ' + ckpt_path
        print('loading from ' + ckpt_path)
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                # print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                # print('module.' + name)
            elif 'model.' + name in ckpt and param.size() == ckpt['model.' + name].size():
                new_params[name].copy_(ckpt['model.' + name])
            # else:
            #     print(name)
        self.model.load_state_dict(new_params)


class HRnet48(BaseBackbone):

    def __init__(self, in_channels):
        super(HRnet48, self).__init__(in_channels)
        from models.backbones.hrnetConfig import config
        # self.model = torchvision.models.resnet50(pretrained=false)
        update_config(config, './config/seg_hrnet_w48.yaml')
        self.model = HighResolutionNet(config)

        # self.model2 = EfficientNet.from_pretrained('efficientnet-b7')

        self.out_channels = [48, 96, 192, 384]
        self.out_channels_sum = sum(self.out_channels)
        self.output_size_num = len(self.out_channels)

        # self.conv1 = nn.conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=false)

        self.load_pretrained_ckpt()

    def forward(self, x):
        x_list = self.model(x)
        return x_list

    def load_pretrained_ckpt(self, device=None):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch
        ckpt_path = './pretrained/HRNET48.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained resnet backbone')
            exit()
        if device is not None:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % device}
            ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        # self.model.load_state_dict(ckpt)

        new_params = self.model.state_dict().copy()
        for name, param in new_params.items():
            if (name in ckpt and param.size() == ckpt[name].size()):
                new_params[name].copy_(ckpt[name])
                # print(name)
            elif 'module.' + name in ckpt and param.size() == ckpt['module.' + name].size():
                new_params[name].copy_(ckpt['module.' + name])
                # print('module.' + name)
            elif 'model.' + name in ckpt and param.size() == ckpt['model.' + name].size():
                new_params[name].copy_(ckpt['model.' + name])
            else:
                print(name)
        self.model.load_state_dict(new_params)
