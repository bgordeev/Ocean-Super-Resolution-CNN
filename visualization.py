"""
Visualization utilities for SST super-resolution.

This module provides functions for visualizing training progress,
comparing model outputs, and creating publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Tuple, List
from pathlib import Path


def visualize_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize a batch of input, target, and output images.
    
    Args:
        inputs: Low-resolution input tensor (N, C, H, W).
        targets: High-resolution target tensor (N, C, H, W).
        outputs: Model output tensor (N, C, H, W).
        num_samples: Number of samples to display.
        save_path: Path to save the figure (optional).
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    num_samples = min(num_samples, inputs.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Convert tensors to numpy and transpose for display
        inp = inputs[i].cpu().detach().permute(1, 2, 0).numpy()
        tgt = targets[i].cpu().detach().permute(1, 2, 0).numpy()
        out = outputs[i].cpu().detach().permute(1, 2, 0).numpy()
        
        # Clip values to [0, 1] range
        inp = np.clip(inp, 0, 1)
        tgt = np.clip(tgt, 0, 1)
        out = np.clip(out, 0, 1)
        
        axes[i, 0].imshow(inp)
        axes[i, 0].set_title('Input (Low-Res)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(tgt)
        axes[i, 1].set_title('Target (High-Res)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(out)
        axes[i, 2].set_title('Output (Model)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_single(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    epoch: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize results for a single sample during training.
    
    Args:
        model: The neural network model.
        input_tensor: Low-resolution input tensor.
        target_tensor: High-resolution target tensor.
        epoch: Current training epoch.
        save_path: Path to save the figure (optional).
    
    Returns:
        Matplotlib figure object.
    """
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
        
        output_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
        input_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        target_img = target_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Clip values
        output_img = np.clip(output_img, 0, 1)
        input_img = np.clip(input_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(input_img)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(target_img)
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    axes[2].imshow(output_img)
    axes[2].set_title('Output')
    axes[2].axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch (optional).
        save_path: Path to save the figure (optional).
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_comparison_grid(
    low_res: np.ndarray,
    high_res: np.ndarray,
    model_output: np.ndarray,
    bilinear: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Create a comparison grid of different resolution methods.
    
    Args:
        low_res: Low-resolution image.
        high_res: High-resolution ground truth.
        model_output: Model's super-resolved output.
        bilinear: Bilinear interpolation result (optional).
        save_path: Path to save the figure (optional).
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    num_cols = 4 if bilinear is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    
    axes[0].imshow(low_res)
    axes[0].set_title(f'Low Resolution\n({low_res.shape[0]}×{low_res.shape[1]})')
    axes[0].axis('off')
    
    col = 1
    if bilinear is not None:
        axes[col].imshow(bilinear)
        axes[col].set_title(f'Bilinear Interpolation\n({bilinear.shape[0]}×{bilinear.shape[1]})')
        axes[col].axis('off')
        col += 1
    
    axes[col].imshow(model_output)
    axes[col].set_title(f'CNN Output\n({model_output.shape[0]}×{model_output.shape[1]})')
    axes[col].axis('off')
    col += 1
    
    axes[col].imshow(high_res)
    axes[col].set_title(f'Ground Truth\n({high_res.shape[0]}×{high_res.shape[1]})')
    axes[col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_sst_map(
    sst_data: np.ndarray,
    title: str = 'Sea Surface Temperature',
    cmap: str = 'RdYlBu_r',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Visualize SST data as a map with colorbar.
    
    Args:
        sst_data: SST data array (2D).
        title: Plot title.
        cmap: Colormap name.
        save_path: Path to save the figure (optional).
        figsize: Figure size.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
    
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mask NaN values
    masked_data = np.ma.masked_invalid(sst_data)
    
    im = ax.imshow(
        masked_data,
        cmap=cmap,
        origin='lower',
        vmin=vmin,
        vmax=vmax
    )
    
    cbar = plt.colorbar(im, ax=ax, label='Temperature (K)')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_map(
    target: np.ndarray,
    prediction: np.ndarray,
    title: str = 'Prediction Error',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot the error between target and prediction.
    
    Args:
        target: Ground truth array.
        prediction: Model prediction array.
        title: Plot title.
        save_path: Path to save the figure (optional).
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    error = target - prediction
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Target
    im0 = axes[0, 0].imshow(target)
    axes[0, 0].set_title('Target')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Prediction
    im1 = axes[0, 1].imshow(prediction)
    axes[0, 1].set_title('Prediction')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Error map
    max_err = np.max(np.abs(error))
    im2 = axes[1, 0].imshow(error, cmap='RdBu', vmin=-max_err, vmax=max_err)
    axes[1, 0].set_title('Error (Target - Prediction)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Absolute error
    im3 = axes[1, 1].imshow(np.abs(error), cmap='hot')
    axes[1, 1].set_title('Absolute Error')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Create dummy data
    low_res = np.random.rand(40, 40, 3)
    high_res = np.random.rand(200, 200, 3)
    
    # Test comparison grid
    fig = create_comparison_grid(
        low_res, high_res, high_res,
        save_path=None
    )
    plt.close(fig)
    
    print("Visualization tests passed!")
