import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class LiteUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(LiteUNet, self).__init__()
        # Encoder: MobileNetV2 (width multiplier 0.35 for fewer channels)&#8203;:contentReference[oaicite:8]{index=8}
        self.encoder = mobilenet_v2(weights=None, width_mult=0.35)
        # Remove MobileNetV2's classifier head (not needed for feature extraction)
        self.encoder.classifier = None

        # Identify channel counts for encoder layers of interest
        # (These correspond to the layers used for skip connections)
        # Using the MobileNetV2 (width_mult=0.35) architecture:
        # Skip connection layers indices: 1, 3, 6, 13 (for 112x112, 56x56, 28x28, 14x14 feature maps)
        # Bottom layer index: 17 (7x7 feature map, just before final conv layer of MobileNetV2)
        enc_channels = []
        for layer in self.encoder.features:
            # For Conv layers or blocks with 'out_channels' attribute, record it
            if hasattr(layer, "out_channels"):
                enc_channels.append(layer.out_channels)
            elif hasattr(layer, "0") and isinstance(layer[0], nn.Conv2d):
                # ConvBNActivation modules (initial conv or final conv in MobileNetV2)
                enc_channels.append(layer[0].out_channels)
        # Channel counts for skip layers and bottom
        c1 = enc_channels[1]  # e.g., 8 channels at 112x112
        c2 = enc_channels[3]  # e.g., 8 channels at 56x56
        c3 = enc_channels[6]  # e.g., 8 channels at 28x28
        c4 = enc_channels[13]  # e.g., 32 channels at 14x14
        c_bottom = enc_channels[17]  # e.g., 112 channels at 7x7 (bottom of encoder)

        # Decoder: Upsampling layers + convolutional blocks with skip connections&#8203;:contentReference[oaicite:9]{index=9}
        # Stage 4 (bottom -> 14x14)
        self.up4 = nn.ConvTranspose2d(c_bottom, c4, kernel_size=2, stride=2)  # upsample 7x7 -> 14x14
        print(c4)
        self.conv4a = nn.Conv2d(c4 + c4, c4, kernel_size=3, padding=1)  # concat channels: c4 (up) + c4 (skip)
        print(c4)

        self.bn4a = nn.BatchNorm2d(c4)
        self.relu4a = nn.ReLU(inplace=True)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, padding=1)
        print(c4)

        self.bn4b = nn.BatchNorm2d(c4)
        self.relu4b = nn.ReLU(inplace=True)

        # Stage 3 (14x14 -> 28x28)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)  # 14x14 -> 28x28
        print(c3)
        self.conv3a = nn.Conv2d(c3 + c3, c3, kernel_size=3, padding=1)  # concat: c3 (up) + c3 (skip)
        print(c3)
        self.bn3a = nn.BatchNorm2d(c3)
        self.relu3a = nn.ReLU(inplace=True)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        print(c3)
        self.bn3b = nn.BatchNorm2d(c3)
        self.relu3b = nn.ReLU(inplace=True)

        # Stage 2 (28x28 -> 56x56)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)  # 28x28 -> 56x56
        print(c2)
        self.conv2a = nn.Conv2d(c2 + c2, c2, kernel_size=3, padding=1)  # concat: c2 (up) + c2 (skip)
        print(c2)
        self.bn2a = nn.BatchNorm2d(c2)
        self.relu2a = nn.ReLU(inplace=True)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        print(c2)
        self.bn2b = nn.BatchNorm2d(c2)
        self.relu2b = nn.ReLU(inplace=True)

        # Stage 1 (56x56 -> 112x112)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)  # 56x56 -> 112x112
        print(c1)
        self.conv1a = nn.Conv2d(c1 + c1, c1, kernel_size=3, padding=1)  # concat: c1 (up) + c1 (skip)
        print(c1)
        self.bn1a = nn.BatchNorm2d(c1)
        self.relu1a = nn.ReLU(inplace=True)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        print(c1)
        self.bn1b = nn.BatchNorm2d(c1)
        self.relu1b = nn.ReLU(inplace=True)

        # Stage 0 (112x112 -> original size 224x224)
        self.up0 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)  # 112x112 -> 224x224
        # Note: We will concatenate the original input (3-channel) here as skip connection
        self.conv0a = nn.Conv2d(c1 + 3, c1, kernel_size=3, padding=1)  # concat: c1 (up) + 3 (input image)
        self.bn0a = nn.BatchNorm2d(c1)
        self.relu0a = nn.ReLU(inplace=True)
        self.conv0b = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.bn0b = nn.BatchNorm2d(c1)
        self.relu0b = nn.ReLU(inplace=True)

        # Final output layer: 1x1 conv to get 1 output channel, then sigmoid&#8203;:contentReference[oaicite:10]{index=10}
        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()  # sigmoid for binary segmentation mask output

    def forward(self, x):
        # Encoder forward pass: collect skip features
        orig = x  # original input for final skip
        skip_feats = []  # to store encoder feature maps for skip connections
        out = x
        for i, layer in enumerate(self.encoder.features):
            out = layer(out)
            if i == 1:  # after 1st inverted residual block (112x112)
                skip_feats.append(out)
            elif i == 3:  # after 2nd inverted residual block (56x56)
                skip_feats.append(out)
            elif i == 6:  # after 3rd inverted residual block (28x28)
                skip_feats.append(out)
            elif i == 13:  # after 4th inverted residual block (14x14)
                skip_feats.append(out)
            if i == 17:  # stop at the bottom layer (7x7)
                bottom = out
                break

        # Decoder forward pass with skip connections
        # Stage 4: 7x7 -> 14x14
        x4 = self.up4(bottom)
        x4 = torch.cat([x4, skip_feats[3]], dim=1)  # concat with encoder 14x14 feature
        x4 = self.relu4a(self.bn4a(self.conv4a(x4)))
        x4 = self.relu4b(self.bn4b(self.conv4b(x4)))
        # Stage 3: 14x14 -> 28x28
        x3 = self.up3(x4)
        x3 = torch.cat([x3, skip_feats[2]], dim=1)  # concat with encoder 28x28 feature
        x3 = self.relu3a(self.bn3a(self.conv3a(x3)))
        x3 = self.relu3b(self.bn3b(self.conv3b(x3)))
        # Stage 2: 28x28 -> 56x56
        x2 = self.up2(x3)
        x2 = torch.cat([x2, skip_feats[1]], dim=1)  # concat with encoder 56x56 feature
        x2 = self.relu2a(self.bn2a(self.conv2a(x2)))
        x2 = self.relu2b(self.bn2b(self.conv2b(x2)))
        # Stage 1: 56x56 -> 112x112
        x1 = self.up1(x2)
        x1 = torch.cat([x1, skip_feats[0]], dim=1)  # concat with encoder 112x112 feature
        x1 = self.relu1a(self.bn1a(self.conv1a(x1)))
        x1 = self.relu1b(self.bn1b(self.conv1b(x1)))
        # Stage 0: 112x112 -> 224x224 (original)
        x0 = self.up0(x1)
        # Concatenate original input as a skip connection (to refine final details)
        if x0.shape[2:] != orig.shape[2:]:
            # If size mismatch due to rounding, interpolate to exact original size
            x0 = nn.functional.interpolate(x0, size=orig.shape[2:], mode='bilinear', align_corners=False)
        x0 = torch.cat([x0, orig], dim=1)
        x0 = self.relu0a(self.bn0a(self.conv0a(x0)))
        x0 = self.relu0b(self.bn0b(self.conv0b(x0)))
        # Output layer
        mask = self.final_conv(x0)
        mask = self.activation(mask)  # sigmoid outputs probability map
        return mask
