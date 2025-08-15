import logging
import torch
from core.models import resnet, vgg
from core.models.feature_extractor import vgg_feature_extractor, resnet_feature_extractor,mit_feature_extractor
from core.models.classifier import ASPP_Classifier_V2,SegFormerHead_Classifier,SAMClassifier
from core.models.discriminator import *
from core.models.GraphBuilder import GraphBuilder
from core.models.vae import GraphVAEWithGraphSAGE

def build_model(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if model_name=='deeplab':
        model = deeplab(backbone_name, cfg.MODEL.NUM_CLASSES, pretrained_weights=cfg.MODEL.WEIGHTS, freeze_bn=cfg.MODEL.FREEZE_BN)
    else:
        raise NotImplementedError
    return model

def build_feature_extractor(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('resnet'):
        backbone = resnet_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=True, pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
    elif backbone_name.startswith('vgg'):
        backbone = vgg_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False, pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
    elif backbone_name.startswith('mitbx'):
        backbone_setting = {
            "pretrain":'/data/zd/dg/DSU-main/HRDA/HRDA-dg/pretrained/mit_b5.pth',
            "variety": "b5",
        }
        backbone = mit_feature_extractor(backbone_setting)
    else:
        raise NotImplementError
    return backbone


def build_classifier(cfg):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('mitbx'):
        decoder = {
            "align_corners":False,
            "channels": 128,
            "num_classes":cfg.MODEL.NUM_CLASSES,
            "in_channels":[64, 128, 320, 512]
        }
        classifier =SAMClassifier(align_corners=decoder["align_corners"], channels=decoder["channels"],
                                             num_classes=decoder["num_classes"],in_channels=decoder["in_channels"])

    else:
        raise NotImplementedError
    return classifier

def build_GraphVAE(cfg):
    graph_vae = GraphVAEWithGraphSAGE(input_dim=128, latent_dim=256, num_classes=cfg.MODEL.NUM_CLASSES,)
    return graph_vae


def build_aux_classifier(cfg):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('vgg'):
        classifier = ASPP_Classifier_V2(512, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('resnet'):
        classifier = ASPP_Classifier_V2(1024, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return classifier

def build_adversarial_discriminator(cfg, num_features=None, mid_nc=256):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('vgg'):
        if num_features is None:
            num_features = 1024
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('resnet'):
        if num_features is None:
            num_features = 2048
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return model_D
