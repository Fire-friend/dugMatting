from .wrapper import *

# ------------------------------------------------------------------------------
#  Replaceable Backbones
# ------------------------------------------------------------------------------

SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
    'mobilenetv2_human': MobileNetV2Backbone_human,
    'mobilenetv3': MobileNetV3Backbone,
    'mobilenetv3_large': MobileNetV3LargeBackbone,
    'mobilenetv3_small': MobileNetV3SmallBackbone,
    'resnet50': ResnetBackbone,
    'resnet34': ResnetBackbone_r34,
    'swin_transformer': SwinT,
    'convnext': Convnext,
    'efficient_b7': Efficient_b7,
    'efficient_b6': Efficient_b6,
    'hrnet18': HRnet18,
    'hrnet48': HRnet48,
    # 'efficient_b5': Efficient_b5,
}
