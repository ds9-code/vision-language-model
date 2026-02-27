import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Tuple, Optional, Union
import math


def visualize_mri_volume(volume: torch.Tensor, slice_idx: Optional[int] = None, 
                         orientation: str = 'axial', title: str = '', 
                         figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Visualize a 3D MRI volume.
    
    Args:
        volume: MRI volume of shape [1, D, H, W]
        slice_idx: Index of slice to visualize. If None, use the middle slice.
        orientation: Orientation of slice ('axial', 'coronal', or 'sagittal')
        title: Title for the plot
        figsize: Figure size
    """
    # Convert to numpy if tensor
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze(0).detach().cpu().numpy()  # Remove channel dim
    
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume after removing channel dim, got shape {volume.shape}")
    
    D, H, W = volume.shape
    
    # Get the appropriate slice
    if orientation.lower() == 'axial':
        if slice_idx is None:
            slice_idx = D // 2
        slc = volume[slice_idx, :, :]
        plane_desc = f"Axial slice {slice_idx}/{D}"
    elif orientation.lower() == 'coronal':
        if slice_idx is None:
            slice_idx = H // 2
        slc = volume[:, slice_idx, :]
        plane_desc = f"Coronal slice {slice_idx}/{H}"
    elif orientation.lower() == 'sagittal':
        if slice_idx is None:
            slice_idx = W // 2
        slc = volume[:, :, slice_idx]
        plane_desc = f"Sagittal slice {slice_idx}/{W}"
    else:
        raise ValueError(f"Unknown orientation: {orientation}")
    
    # Plot the slice
    plt.figure(figsize=figsize)
    plt.imshow(slc, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title(f"{title} - {plane_desc}" if title else plane_desc)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_multi_volumes(volumes: List[torch.Tensor], slice_idxs: Optional[List[int]] = None,
                           orientation: str = 'axial', titles: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 4)) -> None:
    """
    Visualize multiple 3D MRI volumes.
    
    Args:
        volumes: List of MRI volumes, each of shape [1, D, H, W]
        slice_idxs: List of slice indices to visualize. If None, use middle slices.
        orientation: Orientation of slice ('axial', 'coronal', or 'sagittal')
        titles: List of titles for each volume
        figsize: Figure size
    """
    n_volumes = len(volumes)
    
    if slice_idxs is None:
        slice_idxs = [None] * n_volumes
    
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(n_volumes)]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, n_volumes, figsize=figsize)
    if n_volumes == 1:
        axes = [axes]
    
    for i, (volume, slice_idx, title) in enumerate(zip(volumes, slice_idxs, titles)):
        # Convert to numpy if tensor
        if isinstance(volume, torch.Tensor):
            volume = volume.squeeze(0).detach().cpu().numpy()  # Remove channel dim
        
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume after removing channel dim, got shape {volume.shape}")
        
        D, H, W = volume.shape
        
        # Get the appropriate slice
        if orientation.lower() == 'axial':
            if slice_idx is None:
                slice_idx = D // 2
            slc = volume[slice_idx, :, :]
            plane_desc = f"Axial slice {slice_idx}/{D}"
        elif orientation.lower() == 'coronal':
            if slice_idx is None:
                slice_idx = H // 2
            slc = volume[:, slice_idx, :]
            plane_desc = f"Coronal slice {slice_idx}/{H}"
        elif orientation.lower() == 'sagittal':
            if slice_idx is None:
                slice_idx = W // 2
            slc = volume[:, :, slice_idx]
            plane_desc = f"Sagittal slice {slice_idx}/{W}"
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        
        # Plot the slice
        axes[i].imshow(slc, cmap='gray')
        axes[i].set_title(f"{title}\n{plane_desc}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                          image_labels: Optional[List[str]] = None, 
                          text_labels: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot similarity matrix between image and text embeddings.
    
    Args:
        image_embeddings: Image embeddings of shape [n_images, embedding_dim]
        text_embeddings: Text embeddings of shape [n_texts, embedding_dim]
        image_labels: Labels for image embeddings
        text_labels: Labels for text embeddings
        figsize: Figure size
    """
    # Compute similarity matrix
    similarity = torch.matmul(image_embeddings, text_embeddings.T).detach().cpu().numpy()
    
    # Create labels if not provided
    if image_labels is None:
        image_labels = [f"Image {i+1}" for i in range(len(image_embeddings))]
    if text_labels is None:
        text_labels = [f"Text {i+1}" for i in range(len(text_embeddings))]
    
    # Plot similarity matrix
    plt.figure(figsize=figsize)
    sns.heatmap(similarity, annot=True, fmt=".2f", cmap="YlGnBu",
               xticklabels=text_labels, yticklabels=image_labels)
    plt.xlabel("Text Embeddings")
    plt.ylabel("Image Embeddings")
    plt.title("Cosine Similarity Between Image and Text Embeddings")
    plt.tight_layout()
    plt.show()


def plot_training_metrics(train_losses: List[float], val_losses: List[float],
                         train_accs: Optional[List[float]] = None,
                         val_accs: Optional[List[float]] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=figsize)
    
    # Plot losses
    if train_accs is not None:
        ax_loss = axes[0]
    else:
        ax_loss = axes
    
    epochs = range(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax_loss.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax_loss.set_title('Training and Validation Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    
    # Plot accuracies if provided
    if train_accs is not None and val_accs is not None:
        ax_acc = axes[1]
        ax_acc.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax_acc.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax_acc.set_title('Training and Validation Accuracy')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)
    
    plt.tight_layout()
    plt.show() 