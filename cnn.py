"""
CNN model architectures for SST super-resolution.

This module contains neural network architectures for upscaling low-resolution
sea surface temperature data to high resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SuperResolutionCNN(nn.Module):
    """
    Basic CNN for 5× super-resolution of SST images.
    
    This model uses bilinear upsampling followed by convolutional layers
    with skip connections to enhance image resolution.
    
    Architecture:
        Input (3×40×40) → Bilinear Upsample (5×) → Conv layers → Output (3×200×200)
    
    Attributes:
        scale_factor: The upscaling factor (default: 5).
        initial_upsample: Bilinear upsampling layer.
        conv1-4: Convolutional layers.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 5,
        base_filters: int = 64
    ):
        """
        Initialize the super-resolution model.
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB).
            scale_factor: Upscaling factor (default: 5).
            base_filters: Number of filters in first conv layer.
        """
        super(SuperResolutionCNN, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Initial upsampling to target resolution
        self.initial_upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_filters, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
        
        Returns:
            Output tensor of shape (N, C, H*scale, W*scale).
        """
        # Upsample input to target resolution
        x_upsampled = self.initial_upsample(x)
        
        # Convolutional layers with skip connections
        x1 = F.relu(self.conv1(x_upsampled))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2) + x1)  # Skip connection
        x4 = self.conv4(x3) + x_upsampled  # Skip connection from input
        
        return x4


class ResidualBlock(nn.Module):
    """
    Residual block for deeper super-resolution networks.
    
    Consists of two convolutional layers with a skip connection.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Initialize the residual block.
        
        Args:
            channels: Number of input/output channels.
            kernel_size: Size of convolutional kernels.
        """
        super(ResidualBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        
        return F.relu(out)


class EnhancedSuperResolutionCNN(nn.Module):
    """
    Enhanced CNN with residual blocks for better super-resolution.
    
    This architecture uses multiple residual blocks for better feature
    extraction and detail preservation.
    
    Architecture:
        Input → Initial Conv → Residual Blocks → Upsample → Output Conv → Output
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 5,
        base_filters: int = 64,
        num_residual_blocks: int = 8
    ):
        """
        Initialize the enhanced super-resolution model.
        
        Args:
            in_channels: Number of input channels.
            scale_factor: Upscaling factor.
            base_filters: Number of filters in convolutional layers.
            num_residual_blocks: Number of residual blocks.
        """
        super(EnhancedSuperResolutionCNN, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_filters) for _ in range(num_residual_blocks)]
        )
        
        # Post-residual convolution
        self.post_residual = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters)
        )
        
        # Upsampling layers using pixel shuffle
        upsample_layers = []
        current_scale = 1
        
        while current_scale < scale_factor:
            upsample_layers.extend([
                nn.Conv2d(base_filters, base_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
            current_scale *= 2
        
        self.upsample = nn.Sequential(*upsample_layers)
        
        # Final adjustment to match exact output size
        self.final_upsample = nn.Upsample(
            scale_factor=scale_factor / current_scale,
            mode='bilinear',
            align_corners=True
        ) if current_scale != scale_factor else nn.Identity()
        
        # Output convolution
        self.output_conv = nn.Conv2d(base_filters, in_channels, kernel_size=9, padding=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
        
        Returns:
            Output tensor of shape (N, C, H*scale, W*scale).
        """
        # Initial feature extraction
        initial = self.initial_conv(x)
        
        # Residual blocks
        residual = self.residual_blocks(initial)
        residual = self.post_residual(residual)
        
        # Global skip connection
        features = initial + residual
        
        # Upsampling
        upsampled = self.upsample(features)
        upsampled = self.final_upsample(upsampled)
        
        # Output
        output = self.output_conv(upsampled)
        
        return output


class SubPixelCNN(nn.Module):
    """
    Sub-pixel CNN (ESPCN) for efficient super-resolution.
    
    Uses pixel shuffle (sub-pixel convolution) for upsampling, which is
    more efficient than deconvolution or interpolation-based methods.
    
    Reference:
        Shi et al., "Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 5,
        base_filters: int = 64
    ):
        """
        Initialize the sub-pixel CNN.
        
        Args:
            in_channels: Number of input channels.
            scale_factor: Upscaling factor.
            base_filters: Number of filters in hidden layers.
        """
        super(SubPixelCNN, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        
        # Sub-pixel convolution layer
        # Output channels = in_channels * scale_factor^2 for pixel shuffle
        self.conv4 = nn.Conv2d(
            base_filters,
            in_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
        
        Returns:
            Output tensor of shape (N, C, H*scale, W*scale).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        
        return x


def get_model(
    model_name: str = 'basic',
    **kwargs
) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model ('basic', 'enhanced', 'subpixel').
        **kwargs: Additional arguments passed to the model constructor.
    
    Returns:
        Instantiated model.
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    models = {
        'basic': SuperResolutionCNN,
        'enhanced': EnhancedSuperResolutionCNN,
        'subpixel': SubPixelCNN
    }
    
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models.keys())}"
        )
    
    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing model architectures...\n")
    
    # Create sample input
    batch_size = 2
    channels = 3
    low_res_size = 40
    x = torch.randn(batch_size, channels, low_res_size, low_res_size)
    
    # Test each model
    for name in ['basic', 'enhanced', 'subpixel']:
        model = get_model(name)
        output = model(x)
        params = count_parameters(model)
        
        print(f"{name.capitalize()} Model:")
        print(f"  Input shape:  {tuple(x.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Parameters:   {params:,}")
        print()
