import torch

from rfconnectorai.classifier.train import build_model


def test_resnet18_backbone_shape():
    m = build_model(num_classes=9, architecture="resnet18").eval()
    out = m(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 9)


def test_efficientnet_v2_s_backbone_shape():
    m = build_model(num_classes=9, architecture="efficientnet_v2_s").eval()
    out = m(torch.zeros(1, 3, 384, 384))
    assert out.shape == (1, 9)
