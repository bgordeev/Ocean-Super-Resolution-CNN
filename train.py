"""
Training script for SST super-resolution model.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Training loop with checkpointing
- Visualization of results
"""

import os
import argparse
import glob
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.cnn import get_model, count_parameters
from utils.visualization import visualize_single, plot_training_history


def load_image(file_path: str) -> np.ndarray:
    """
    Load an image file and convert to numpy array.
    
    Args:
        file_path: Path to the image file.
    
    Returns:
        Numpy array of shape (H, W, C).
    """
    img = Image.open(file_path)
    img = img.convert('RGB')
    return np.array(img)


def load_dataset(
    low_res_dir: str,
    high_res_dir: str,
    pattern: str = "*/*.png"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load low and high resolution image pairs.
    
    Args:
        low_res_dir: Directory containing low-resolution images.
        high_res_dir: Directory containing high-resolution images.
        pattern: Glob pattern for finding image files.
    
    Returns:
        Tuple of (low_res_tensor, high_res_tensor).
    """
    print("Loading file paths...")
    low_res_paths = sorted(glob.glob(os.path.join(low_res_dir, pattern)))
    high_res_paths = sorted(glob.glob(os.path.join(high_res_dir, pattern)))
    
    if len(low_res_paths) != len(high_res_paths):
        raise ValueError(
            f"Mismatch in number of files: {len(low_res_paths)} low-res, "
            f"{len(high_res_paths)} high-res"
        )
    
    if len(low_res_paths) == 0:
        raise ValueError(f"No image files found in {low_res_dir}")
    
    print(f"Found {len(low_res_paths)} image pairs")
    
    print("Processing images...")
    low_res_imgs = [load_image(p) for p in tqdm(low_res_paths, desc="Low-res")]
    high_res_imgs = [load_image(p) for p in tqdm(high_res_paths, desc="High-res")]
    
    print("Converting to tensors...")
    # Convert to tensors: (N, H, W, C) -> (N, C, H, W) and normalize to [0, 1]
    x_data = torch.tensor(
        [img.transpose((2, 0, 1)) for img in low_res_imgs],
        dtype=torch.float32
    ) / 255.0
    
    y_data = torch.tensor(
        [img.transpose((2, 0, 1)) for img in high_res_imgs],
        dtype=torch.float32
    ) / 255.0
    
    print(f"Low-res shape: {x_data.shape}")
    print(f"High-res shape: {y_data.shape}")
    
    return x_data, y_data


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
    
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for x_batch, y_batch in progress_bar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model.
        test_loader: DataLoader for test data.
        criterion: Loss function.
        device: Device to evaluate on.
    
    Returns:
        Average loss on test set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    checkpoint_dir: str,
    device: torch.device,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    visualize_every: int = 1
) -> Tuple[list, list]:
    """
    Full training loop.
    
    Args:
        model: Neural network model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        checkpoint_dir: Directory to save checkpoints.
        device: Device to train on.
        x_test: Test input tensor for visualization.
        y_test: Test target tensor for visualization.
        visualize_every: Visualize results every N epochs.
    
    Returns:
        Tuple of (train_losses, test_losses).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        test_loss = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Test Loss:  {test_loss:.6f}")
        
        # Save checkpoint
        checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, checkpoint_file)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_file = checkpoint_path / "best_model.pth"
            torch.save(model.state_dict(), best_file)
            print(f"New best model saved (loss: {best_loss:.6f})")
        
        # Visualize results
        if epoch % visualize_every == 0:
            vis_file = checkpoint_path / f"visualization_epoch_{epoch}.png"
            model.to('cpu')  # Move to CPU for visualization
            visualize_single(
                model,
                x_test[0].unsqueeze(0),
                y_test[0].unsqueeze(0),
                epoch,
                save_path=str(vis_file)
            )
            model.to(device)
    
    # Save training history plot
    history_file = checkpoint_path / "training_history.png"
    plot_training_history(train_losses, test_losses, save_path=str(history_file))
    
    return train_losses, test_losses


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train SST super-resolution model'
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data/patches',
        help='Directory containing patch data'
    )
    parser.add_argument(
        '--checkpoint-dir', '-c',
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Fraction of data for testing'
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
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("SST Super-Resolution Training")
    print("=" * 60)
    
    # Load data
    low_res_dir = os.path.join(args.data_dir, 'low_res')
    high_res_dir = os.path.join(args.data_dir, 'high_res')
    
    x_data, y_data = load_dataset(low_res_dir, high_res_dir)
    
    # Split data
    print("\nSplitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data,
        test_size=args.test_split,
        random_state=args.seed
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples:     {len(x_test)}")
    
    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model)
    model = model.to(device)
    print(model)
    
    # Train
    train_losses, test_losses = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        x_test=x_test,
        y_test=y_test
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best test loss: {min(test_losses):.6f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
