#
# model.py
#
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. The Differentiable Bottleneck Layers (from before)
# -----------------------------------------------------------------------------

class SpatialSoftmax(nn.Module):
    """
    Converts a heatmap (B, K, H, W) to a set of (x, y) coordinates (B, K, 2).
    """
    def __init__(self, height, width, temperature=1.0):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.temperature = temperature

        x_coords = torch.linspace(0, 1, width)
        y_coords = torch.linspace(0, 1, height)
        y_map, x_map = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        self.register_buffer('x_map', x_map.unsqueeze(0).unsqueeze(0))
        self.register_buffer('y_map', y_map.unsqueeze(0).unsqueeze(0))

    def forward(self, heatmaps):
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, K, -1)
        softmax_probs = F.softmax(heatmaps_flat / self.temperature, dim=2)
        softmax_probs = softmax_probs.view(B, K, H, W)

        expected_x = torch.sum(softmax_probs * self.x_map, dim=[2, 3])
        expected_y = torch.sum(softmax_probs * self.y_map, dim=[2, 3])
        
        keypoints_xy = torch.stack([expected_x, expected_y], dim=2)
        return keypoints_xy

class GaussianRenderer(nn.Module):
    """
    Converts (x, y) coordinates (B, K, 2) back into heatmaps (B, K, H, W).
    """
    def __init__(self, height, width, sigma=0.05):
        super(GaussianRenderer, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma

        x_coords = torch.linspace(0, 1, width)
        y_coords = torch.linspace(0, 1, height)
        y_map, x_map = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        self.register_buffer('x_map', x_map.unsqueeze(0).unsqueeze(0))
        self.register_buffer('y_map', y_map.unsqueeze(0).unsqueeze(0))

    def forward(self, keypoints_xy):
        B, K, _ = keypoints_xy.shape
        mu_x = keypoints_xy[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        mu_y = keypoints_xy[:, :, 1].unsqueeze(-1).unsqueeze(-1)

        x_diff = self.x_map - mu_x
        y_diff = self.y_map - mu_y
        
        variance = self.sigma ** 2
        exponent = -((x_diff ** 2) + (y_diff ** 2)) / (2 * variance)
        heatmaps = torch.exp(exponent)
        
        return heatmaps

# -----------------------------------------------------------------------------
# 2. The U-Net Decoder Architecture
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool then ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 is the skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -----------------------------------------------------------------------------
# 3. The Final Assembled Model
# -----------------------------------------------------------------------------

class PKA_UNet(nn.Module):
    """
    Keypoint Autoencoder with a U-Net Decoder.
    
    Encoder: Simple CNN
    Bottleneck: SpatialSoftmax -> (x,y) Coords -> GaussianRenderer
    Decoder: U-Net (takes Gaussian maps as input)
    """
    def __init__(self, n_channels_in=3, n_channels_out=3, num_keypoints=10, img_size=128):
        super(PKA_UNet, self).__init__()
        self.num_keypoints = num_keypoints
        self.img_size = img_size
        
        # --- 1. ENCODER (Image -> Heatmaps) ---
        # A simple ResNet-like encoder
        self.enc_in = ConvBlock(n_channels_in, 64)
        self.enc_d1 = Down(64, 128)
        self.enc_d2 = Down(128, 256)
        self.enc_d3 = Down(256, 512)
        # Final conv to get K heatmaps at 1/8th resolution
        self.heatmap_conv = nn.Conv2d(512, num_keypoints, kernel_size=1)
        # (B, K, 16, 16) for a 128 input
        
        # --- 2. BOTTLENECK (Heatmaps -> Coords -> Gaussian Maps) ---
        self.spatial_softmax = SpatialSoftmax(height=img_size // 8, width=img_size // 8)
        self.gaussian_renderer = GaussianRenderer(height=img_size, width=img_size)
        
        # --- 3. DECODER (Gaussian Maps -> Reconstructed Image) ---
        # This is a U-Net that takes the K-channel Gaussian map as input
        self.dec_in = ConvBlock(num_keypoints, 64)
        self.dec_d1 = Down(64, 128)
        self.dec_d2 = Down(128, 256)
        self.dec_d3 = Down(256, 512)
        self.dec_d4 = Down(512, 1024)
        
        self.dec_u1 = Up(1024, 512)
        self.dec_u2 = Up(512, 256)
        self.dec_u3 = Up(256, 128)
        self.dec_u4 = Up(128, 64)
        self.dec_out = nn.Conv2d(64, n_channels_out, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- Encoder ---
        x1 = self.enc_in(x)
        x2 = self.enc_d1(x1)
        x3 = self.enc_d2(x2)
        x4 = self.enc_d3(x3)
        heatmaps = self.heatmap_conv(x4)
        
        # --- Bottleneck ---
        keypoints_xy = self.spatial_softmax(heatmaps)
        gaussian_maps = self.gaussian_renderer(keypoints_xy)
        
        # --- Decoder (U-Net) ---
        d1 = self.dec_in(gaussian_maps)
        d2 = self.dec_d1(d1)
        d3 = self.dec_d2(d2)
        d4 = self.dec_d3(d3)
        d5 = self.dec_d4(d4)
        
        u = self.dec_u1(d5, d4)
        u = self.dec_u2(u, d3)
        u = self.dec_u3(u, d2)
        u = self.dec_u4(u, d1)
        
        logits = self.dec_out(u)
        reconstructed_image = self.sigmoid(logits)
        
        return reconstructed_image, keypoints_xy, gaussian_maps
