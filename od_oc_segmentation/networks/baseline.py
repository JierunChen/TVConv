from .baseline_utils import IntermediateLayerGetter
from .baseline_deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .baseline_mobilenetv2 import mobilenet_v2


def _segm_mobilenet(atom, name, backbone_name, num_classes, output_stride, pretrained_backbone, **kwargs):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(atom, inplanes, low_level_planes, num_classes, aspp_dilate, **kwargs)
    elif name == 'deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, **kwargs)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(atom, arch_type, backbone, num_classes, output_stride, pretrained_backbone, **kwargs):
    if backbone == 'mobilenetv2':
        model = _segm_mobilenet(atom, arch_type, backbone, num_classes, output_stride=output_stride,
                                pretrained_backbone=pretrained_backbone, **kwargs)
    else:
        raise NotImplementedError
    return model



def deeplabv3plus_mobilenet(atom, num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(atom, 'deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, **kwargs)