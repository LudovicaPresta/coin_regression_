import torch
import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v3_small, ResNet18_Weights, MobileNet_V3_Small_Weights

def get_backbone(name: str, pretrained=True, grayscale=False):
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if grayscale:
            original_conv = model.conv1
            new_conv = nn.Conv2d(1, original_conv.out_channels,
                                 kernel_size=original_conv.kernel_size,
                                 stride=original_conv.stride,
                                 padding=original_conv.padding,
                                 bias=original_conv.bias is not None)
            with torch.no_grad():
                new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
            model.conv1 = new_conv
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 512

    elif name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        if grayscale:
            original_conv = model.features[0][0]
            new_conv = nn.Conv2d(1, original_conv.out_channels,
                                 kernel_size=original_conv.kernel_size,
                                 stride=original_conv.stride,
                                 padding=original_conv.padding,
                                 bias=original_conv.bias is not None)
            with torch.no_grad():
                new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
            model.features[0][0] = new_conv
        feature_extractor = model.features
        feature_dim = 576

    else:
        raise ValueError(f"Unsupported model name: {name}")

    return feature_extractor, feature_dim


class DualImageRegressor(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True,
                 proj_dim=32, shared_backbone=True, grayscale=True):
        super().__init__()
        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.cnn, feature_dim = get_backbone(backbone_name, pretrained, grayscale)
        else:
            self.front_cnn, feature_dim = get_backbone(backbone_name, pretrained, grayscale)
            self.back_cnn, _ = get_backbone(backbone_name, pretrained, grayscale)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.front_proj = nn.Linear(feature_dim, proj_dim)
        self.back_proj = nn.Linear(feature_dim, proj_dim)

        self.regressor = nn.Sequential(
            nn.Linear(proj_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, front_img, back_img):
        if self.shared_backbone:
            front_feat = self.global_pool(self.cnn(front_img)).view(front_img.size(0), -1)
            back_feat = self.global_pool(self.cnn(back_img)).view(back_img.size(0), -1)
        else:
            front_feat = self.global_pool(self.front_cnn(front_img)).view(front_img.size(0), -1)
            back_feat = self.global_pool(self.back_cnn(back_img)).view(back_img.size(0), -1)

        front_proj = self.front_proj(front_feat)
        back_proj = self.back_proj(back_feat)
        combined = torch.cat((front_proj, back_proj), dim=1)
        return self.regressor(combined)
