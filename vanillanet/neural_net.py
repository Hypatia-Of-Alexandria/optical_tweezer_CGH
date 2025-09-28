import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .device import get_device
from .model import VanillaModel
from .utils import w_theta_grid, calculate_complex_ft, inverse_complex_ft
from .metrics import calculate_inefficiency, calculate_non_uniformity, get_intensity_normalised_from_amplitude, calculate_holography_metrics
from .debug import debug_real_data_detailed

__all__ = [
    'evaluate_model',
    'train_model',
    'visualize_results_with_metrics',
    'load_model'
]


@torch.no_grad()
def evaluate_model(
    model: VanillaModel,
    val_loader: DataLoader,
    var_reg: float,
    device: torch.device,
) -> float:
    """
    Evaluate model on validation set.
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        var_reg: Variance regularization weight
        device: Device to run on
    
    Returns:
        Average validation loss (over validation set)
    """
    model.eval()
    total_loss = 0.0
    total_inefficiency = 0.0
    total_non_uniformity = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)  # shape: (batchsize, 1, size, size)
            batch_size = batch.size(0)
            n_samples += batch_size
            
            # Forward pass
            model_out = model(batch)
            w_i = model_out[:, :, 0]  # amplitude
            theta_i = model_out[:, :, 1]  # phase
            
            # Convert to grid
            w_grid, theta_grid = w_theta_grid(w_i, theta_i, batch)
            
            # Reconstruct
            _, phi_recon = inverse_complex_ft(w_grid, theta_grid)
            A_const = torch.ones_like(batch)
            recon, _ = calculate_complex_ft(A_const, phi_recon) 
            
            # Calculate loss 
            inefficiency_loss = calculate_inefficiency(recon, batch)  # mean over batch
            non_uniformity_loss = calculate_non_uniformity(recon, batch)  # mean over batch
            total_loss_batch = inefficiency_loss + var_reg * non_uniformity_loss 
            
            # Convert batch means -> sums over samples in batch
            total_loss += total_loss_batch.item() * batch_size
            total_inefficiency += inefficiency_loss.item() * batch_size
            total_non_uniformity += non_uniformity_loss.item() * batch_size

        # Normalize by total number of validation samples
        avg_total = total_loss / n_samples
        avg_inefficiency = total_inefficiency / n_samples
        avg_non_uniformity = total_non_uniformity / n_samples
        
        print(f"  Val breakdown: Inefficiency={avg_inefficiency:.6f}, "
              f"Non-uniformity={avg_non_uniformity:.6f}, Total={avg_total:.6f}")
        
        return avg_total


def train_model(
    num_epochs: int = 5,
    lr: float = 1e-3,
    size: int = 64,
    batch_size: int = 32,
    var_reg: float = 100000000,
    val_split: float = 0.2,
    dataset = None,
    patience: int = 5,
    max_grad_norm: float = 1.0,
) -> Tuple[VanillaModel, torch.device, object, list, list]:
    """
    Train the VanillaModel with early stopping and progress tracking.
    
    Args:
        num_epochs: Number of training epochs
        lr: Learning rate
        size: Image size
        batch_size: Batch size
        var_reg: Variance regularization weight
        val_split: Validation split ratio
        dataset: Training dataset (if None, creates default dataset)
        patience: Early stopping patience
        max_grad_norm: Gradient clipping norm
    
    Returns:
        Tuple of (best_model, device, dataset, train_losses, val_losses)
    """
    device = get_device()
    
    # Create dataset if not provided
    if dataset is None:
        from .data import DiscretePointsDataset
        dataset = DiscretePointsDataset(
            num_samples=1000, 
            size=64, 
            min_points=5, 
            max_points=15, 
            min_distance=5, 
            border=4
        )
    
    # Create dataset and dataloader for 64x64 images.
    max_points = dataset.max_points  # Use the maximum points from dataset
    
    # Split dataset into train and validation
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Create model with original architecture
    model = VanillaModel(size=size, max_points=max_points, h1=256, h2=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler - reduce LR by factor of 0.5 every 3 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}")
        running_loss = 0.0
        n_train_samples = 0

        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            curr_batch_size = batch.size(0)
            n_train_samples += curr_batch_size
            
            # Reset gradients
            optimizer.zero_grad()

            model_out = model(batch)  # shape (batch_size, num_points, 2)

            # Extract theta_i, w_i predicted by model
            w_i = model_out[:, :, 0]  # shape: (batchsize, num_points)
            theta_i = model_out[:, :, 1]  # shape: (batchsize, num_points)

            w_grid, theta_grid = w_theta_grid(w_i, theta_i, batch)

            # Inverse fourier transform
            _, arg = inverse_complex_ft(w_grid, theta_grid)  # shape: (batch_size, 1, size, size)

            # Forward fourier transform 
            A = torch.ones_like(batch)  # shape: (batchsize, 1, size, size)
            recon, _ = calculate_complex_ft(A, arg)  # shape: (batchsize, 1, size, size)
            assert torch.isfinite(recon).all(), "recon has NaNs or Infs!"
            
            # Holography-based loss function
            inefficiency_loss = calculate_inefficiency(recon, batch)
            non_uniformity_loss = calculate_non_uniformity(recon, batch)
            total_loss = inefficiency_loss + var_reg * non_uniformity_loss
        
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            running_loss += total_loss.item() * curr_batch_size
        
        # Calculate average training loss
        avg_train_loss = running_loss / n_train_samples
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = evaluate_model(model, val_loader, var_reg, device)
        val_losses.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epochs")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
        
        # Step the scheduler
        scheduler.step()
    
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    return model, device, dataset, train_losses, val_losses


def visualize_results_with_metrics(model, dataset, device, num_samples=5, debug=False):
    """
    Visualize model predictions with holography metrics for each sample.
    FIXED: Proper device handling for consistency.
    
    This function creates a grid of plots showing:
      - Input target magnitude spectrum (T),
      - Predicted phase (θ),
      - Reconstructed magnitude (T_recon),
      - Absolute difference between T and T_recon with metrics.
    """
    
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Get one sample and add batch dimension:
        T = dataset[idx].unsqueeze(0).to(device)  # T shape, (1, 1, 64, 64)

        n_points = T.sum()
        n_points = n_points.cpu().detach().numpy()

        with torch.no_grad():
            model_out = model(T)
            w_i = model_out[:, :, 0] # shape: (num_samples, num_points)
            theta_i = model_out[:, :, 1] # shape: (num_samples, num_points)
            w_grid, theta_grid = w_theta_grid(w_i, theta_i, T)
            _, phi_recon = inverse_complex_ft(w_grid, theta_grid)
            A_const = torch.ones_like(T)
            T_recon, _ = calculate_complex_ft(A_const, phi_recon)

        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUGGING SAMPLE {i+1} (Index {idx})")
            print(f"{'='*60}")
            debug_real_data_detailed(T_recon, T)
            print(f"{'='*60}")

        # Move tensors to CPU and convert to numpy arrays FIRST
        T_np = T.cpu().squeeze().numpy()
        n_points = T.sum().cpu().detach().numpy()
        T_recon_np = T_recon.cpu().squeeze().numpy()
        theta_pred_np = theta_grid.cpu().squeeze().numpy()

        # Now do all the intensity calculations using the consistent function
        # Convert numpy arrays back to torch tensors with proper shape (B, 1, H, W)
        T_torch = torch.from_numpy(T_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        T_recon_torch = torch.from_numpy(T_recon_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Use the consistent function to get normalized intensities
        T_np_int_normalized = get_intensity_normalised_from_amplitude(T_torch).squeeze().numpy()
        T_recon_np_int_normalized = get_intensity_normalised_from_amplitude(T_recon_torch).squeeze().numpy()

        diff_np = T_recon_np_int_normalized - T_np_int_normalized
        
        metrics = calculate_holography_metrics(T_recon_torch, T_torch)
        inefficiency = metrics['inefficiency'].item()
        non_uniformity = metrics['non_uniformity'].item()
        intensity_error = metrics['intensity_error'].item()

        # Plot input normalised intensities T:
        im0 = axs[i, 0].imshow(T_np_int_normalized, cmap='gray')
        axs[i, 0].set_title(f"Input intensities |B|^2 \nPoints: {int(n_points)}")
        axs[i, 0].axis('off')
        plt.colorbar(im0, ax=axs[i, 0])

        # Plot predicted phase θ using HSV colormap:
        im1 = axs[i, 1].imshow(theta_pred_np, cmap='hsv')
        axs[i, 1].set_title("Predicted θ (Phase)")
        axs[i, 1].axis('off')
        plt.colorbar(im1, ax=axs[i, 1])

        # Plot reconstructed intensities |B_recon|^2 normalised to sum = 1:
        im2 = axs[i, 2].imshow(T_recon_np_int_normalized, cmap='gray')
        axs[i, 2].set_title("|B_recon|^2")
        axs[i, 2].axis('off')
        plt.colorbar(im2, ax=axs[i, 2])

        # Plot the absolute difference with metrics:
        im3 = axs[i, 3].imshow(diff_np, cmap='hot')
        axs[i, 3].set_title(f"Difference |B|^2 - |B_recon|^2\n"
                           f"Ineff: {inefficiency:.3f}\n"
                           f"Non-unif: {non_uniformity:.3f}\n"
                           f"Intensity err: {intensity_error:.3f}")
        axs[i, 3].axis('off')
        plt.colorbar(im3, ax=axs[i, 3])

    plt.suptitle('Neural Network Predictions with Metrics', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('images/neural_network_predictions_with_metrics.png', bbox_inches='tight')
    plt.show()


def load_model(size: int, max_points: int, h1: int, h2: int, ckpt_path: str, device: torch.device = None) -> VanillaModel:
    """Load a trained model from checkpoint."""
    if device is None:
        device = get_device()
    model = VanillaModel(size=size, max_points=max_points, h1=h1, h2=h2).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model