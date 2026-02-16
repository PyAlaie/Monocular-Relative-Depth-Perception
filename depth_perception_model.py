import torch.nn as nn
import torch

def resnet_encoder():
    """
        Load resnext101 and use first 4 pretrained layers as encoder,
        output channels are 256, 512, 1024, 2048 for each layer accordingly
    """

    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")

    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def create_unet_skip_connections(resnet):
    connections = nn.Module()

    connections.layer1_rn = nn.Conv2d(
        256, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1
    )
    connections.layer2_rn = nn.Conv2d(
        512, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1
    )
    connections.layer3_rn = nn.Conv2d(
        1024, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1
    )
    connections.layer4_rn = nn.Conv2d(
        2048, 256, kernel_size=3, stride=1, padding=1, bias=False, groups=1
    )

    return connections


class ResudualConvBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResudualConvBlock(features)
        self.resConfUnit2 = ResudualConvBlock(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()

        self.resnet_encoder = resnet_encoder()
        self.skip_connections = create_unet_skip_connections(self.resnet_encoder)

        self.feature_fusion1 = FeatureFusionBlock(256)
        self.feature_fusion2 = FeatureFusionBlock(256)
        self.feature_fusion3 = FeatureFusionBlock(256)
        self.feature_fusion4 = FeatureFusionBlock(256)

        self.adaptive_output = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        Interpolate(scale_factor=2, mode="bilinear"),
        nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # Encoding part
        encoded_layer_1 = self.resnet_encoder.layer1(x)
        encoded_layer_2 = self.resnet_encoder.layer2(encoded_layer_1)
        encoded_layer_3 = self.resnet_encoder.layer3(encoded_layer_2)
        encoded_layer_4 = self.resnet_encoder.layer4(encoded_layer_3)

        # Skip connections from encoder
        skip1 = self.skip_connections.layer1_rn(encoded_layer_1)
        skip2 = self.skip_connections.layer2_rn(encoded_layer_2)
        skip3 = self.skip_connections.layer3_rn(encoded_layer_3)
        skip4 = self.skip_connections.layer4_rn(encoded_layer_4)

        # Fusing skip connections into the decoder part
        decoded_layer4 = self.feature_fusion4(skip4)
        decoded_layer3 = self.feature_fusion3(skip3, decoded_layer4)
        decoded_layer2 = self.feature_fusion2(skip2, decoded_layer3)
        decoded_layer1 = self.feature_fusion1(skip1, decoded_layer2)

        # Adaptive output part
        final_result = self.adaptive_output(decoded_layer1)

        return final_result