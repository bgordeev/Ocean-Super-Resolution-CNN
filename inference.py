"""
Inference script for SST super-resolution model.

This script loads a trained model and performs super-resolution
on input images.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.cnn import get_model


def load_model(
    checkpoint_path: str,
    model_name: str = 'basic',
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint.
        model_name: Name of the model architecture.
        device: Device to load the model on.
    
    Returns:
        Loaded model in evaluation mode.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(model_name)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the input image.
    
    Returns:
        Preprocessed tensor of shape (1, C, H, W).
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor: (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(img_array.transpose((2, 0, 1)))
    tensor = tensor.unsqueeze(0)
    
    return tensor


def postprocess_output(output_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert model output tensor to image array.
    
    Args:
        output_tensor: Model output tensor of shape (1, C, H, W).
    
    Returns:
        Image array of shape (H, W, C) with values in [0, 255].
    """
    # Remove batch dimension and convert to numpy
    output = output_tensor.squeeze(0).cpu().detach().numpy()
    
    # Transpose: (C, H, W) -> (H, W, C)
    output = output.transpose((1, 2, 0))
    
    # Clip and convert to uint8
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    return output


def super_resolve(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Apply super-resolution to an input image.
    
    Args:
        model: Trained super-resolution model.
        input_image: Input tensor of shape (1, C, H, W).
        device: Device to run inference on.
    
    Returns:
        Output tensor of shape (1, C, H*scale, W*scale).
    """
    with torch.no_grad():
        input_image = input_image.to(device)
        output = model(input_image)
    
    return output


def process_single_image(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: torch.device,
    visualize: bool = False
) -> None:
    """
    Process a single image through the model.
    
    Args:
        model: Trained super-resolution model.
        input_path: Path to input image.
        output_path: Path to save output image.
        device: Device to run inference on.
        visualize: Whether to display the results.
    """
    # Load and preprocess
    input_tensor = preprocess_image(input_path)
    
    # Super-resolve
    output_tensor = super_resolve(model, input_tensor, device)
    
    # Postprocess and save
    output_array = postprocess_output(output_tensor)
    output_image = Image.fromarray(output_array)
    output_image.save(output_path)
    
    print(f"Saved: {output_path}")
    
    if visualize:
        # Load original for comparison
        input_array = np.array(Image.open(input_path).convert('RGB'))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(input_array)
        axes[0].set_title(f'Input ({input_array.shape[1]}×{input_array.shape[0]})')
        axes[0].axis('off')
        
        axes[1].imshow(output_array)
        axes[1].set_title(f'Output ({output_array.shape[1]}×{output_array.shape[0]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def process_directory(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    pattern: str = "*.png"
) -> None:
    """
    Process all images in a directory.
    
    Args:
        model: Trained super-resolution model.
        input_dir: Directory containing input images.
        output_dir: Directory to save output images.
        device: Device to run inference on.
        pattern: Glob pattern for finding images.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob(pattern))
    
    if len(image_files) == 0:
        print(f"No images found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for img_file in image_files:
        output_file = output_path / f"sr_{img_file.name}"
        process_single_image(model, str(img_file), str(output_file), device)
    
    print(f"Done! Output saved to {output_dir}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Run SST super-resolution inference'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input image path or directory'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output image path or directory'
    )
    parser.add_argument(
        '--model',
        choices=['basic', 'enhanced', 'subpixel'],
        default='basic',
        help='Model architecture'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device (auto, cpu, cuda, or cuda:N)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Display visualization (single image only)'
    )
    parser.add_argument(
        '--pattern',
        default='*.png',
        help='Glob pattern for directory processing'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, device)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        process_single_image(
            model,
            args.input,
            args.output,
            device,
            visualize=args.visualize
        )
    elif input_path.is_dir():
        # Directory
        process_directory(
            model,
            args.input,
            args.output,
            device,
            pattern=args.pattern
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
