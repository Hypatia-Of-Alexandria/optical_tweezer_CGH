import torch
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from typing import Dict, List, Tuple, Optional

from .utils import w_theta_grid_2d, fftshift, ifftshift
from .metrics import calculate_inefficiency, calculate_non_uniformity, _normalize_intensity, calculate_holography_metrics


__all__ = [
    'train_gd_optimization',
    'visualize_gd_results_with_metrics'
]

def train_gd_optimization(
    target_sample=None, 
    max_points=15, 
    iterations=20000, 
    lr=0.01, 
    var_cost=1000, 
    momentum=0.9,
    dataset_size=100, 
    size=64, 
    min_points=5, 
    max_points_dataset=15,
    min_distance=5, 
    border=4, 
    target_index=0, 
    visualize=True, 
    save_path='images/GD_sample_predictions.png',
    show_metrics=True, 
    verbose=True, 
    device=None, 
    show_loss_plot=True
) -> Dict:
    """
    GD optimization using Adam optimizer and holography-based loss function.
    Based on working code from Woj_Test_Sparse_ONLY.ipynb
    stop training after number of iterations = iterations
    
    Args:
        target_sample (torch.Tensor): Target pattern, shape (1, H, W) or (H, W)
        max_points (int): Maximum number of point sources
        iterations (int): Number of optimization iterations
        lr (float): Learning rate for SGD optimizer
        var_cost (float): Weight for non-uniformity loss
        visualize (bool): Create visualization plots
        show_metrics (bool): Display metrics
        verbose (bool): Print detailed progress
        device: Computing device
        show_loss_plot (bool): Show loss curve plot
        
    Returns:
        dict: Results with optimized parameters and loss history
    """
    from .data import DiscretePointsDataset
    
    # Determine device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset or use provided target
    if target_sample is None:
        dataset = DiscretePointsDataset(
            num_samples=dataset_size, size=size, 
            min_points=min_points, max_points=max_points_dataset, 
            min_distance=min_distance, border=border
        )
        target_sample = dataset[target_index]  # Shape: (1, 64, 64)
    
    # Ensure target is 2D for w_theta_grid_2d
    if target_sample.dim() == 3:
        target_binary = target_sample.squeeze()  # Shape: (64, 64)
    else:
        target_binary = target_sample
    
    target_binary = target_binary.to(device)
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Target shape: {target_binary.shape}")
        print(f"Number of points in target: {target_binary.sum().item()}")
        print(f"Starting GD optimization with {iterations} iterations...")
    
    # Initialize learnable parameters
    learnable_w = torch.ones((1, max_points), requires_grad=True, device=device)
    learnable_theta = torch.randn((1, max_points), requires_grad=True, device=device)
    
    # Setup GD optimizer and scheduler
    optimizer = torch.optim.Adam([learnable_w, learnable_theta], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000, verbose=verbose
    )
    
    # Track losses and time
    losses = []
    start_time = perf_counter()
    
    # Training loop
    for i in range(iterations):
        optimizer.zero_grad()

        # w_theta_grid_2d expects 2D target
        w_grid, theta_grid = w_theta_grid_2d(learnable_w, learnable_theta, target_binary)
        complex_grid = w_grid * torch.exp(1j * theta_grid)
        spatial = torch.fft.ifft2(ifftshift(complex_grid, dim=(-2, -1)))
        mag = torch.sqrt((spatial.real**2 + spatial.imag**2) + 1e-12)
        learnable_array = spatial / mag  # unit-modulus, keeps phase without angle()

        predicted_ft = fftshift(torch.fft.fft2(learnable_array), dim=(-2, -1))
        predicted_intensity = (predicted_ft.conj() * predicted_ft).real
        predicted_amplitude = torch.sqrt(predicted_intensity + 1e-12)
        
        # Add batch and channel dimensions for metrics functions
        predicted_amplitude_batch = predicted_amplitude.unsqueeze(0).unsqueeze(0)
        target_binary_batch = target_binary.unsqueeze(0).unsqueeze(0)
        
        # Calculate inefficiency and non-uniformity
        inefficiency = calculate_inefficiency(predicted_amplitude_batch, target_binary_batch)
        non_uniformity = calculate_non_uniformity(predicted_amplitude_batch, target_binary_batch)

        # # DEBUG: Check if gradients are preserved
        # if i == 0:
        #     print(f"Target points: {target_binary.sum().item()}")
        #     print(f"Inefficiency requires_grad: {inefficiency.requires_grad}")
        #     print(f"Non_uniformity requires_grad: {non_uniformity.requires_grad}")
        
        # Weighted combination of losses
        loss = inefficiency + var_cost * non_uniformity
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Track loss
        losses.append(loss.item())
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}, Inefficiency: {inefficiency.item():.6f}, Non-uniformity: {non_uniformity.item():.6f}")
    
    end_time = perf_counter()
    reconstruction_time = end_time - start_time
    
    if verbose:
        print("Optimization complete!")
        print(f"Final loss: {loss.item():.6f}")
        print(f"Total time: {reconstruction_time:.2f} seconds")
    
    # Plot loss curve
    if show_loss_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(losses, 'b-', linewidth=1)
        plt.title('GD Loss vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/gd_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f}")
        print(f"Total improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")

    # Calculate final reconstruction (always needed for results)
    with torch.no_grad():
        final_w_grid, final_theta_grid = w_theta_grid_2d(learnable_w, learnable_theta, target_binary)
        final_complex_grid = final_w_grid * torch.exp(1j * final_theta_grid)
        
        # Use angle-free reconstruction (consistent with training)
        final_spatial = torch.fft.ifft2(ifftshift(final_complex_grid, dim=(-2, -1)))
        final_mag = torch.sqrt((final_spatial.real**2 + final_spatial.imag**2) + 1e-12)
        final_learnable_array = final_spatial / final_mag
        
        final_predicted_ft = fftshift(torch.fft.fft2(final_learnable_array))
        final_predicted_amplitude = torch.abs(final_predicted_ft)

    # Calculate final metrics (optional)
    final_metrics = {}
    if show_metrics:
        final_predicted_amplitude_batch = final_predicted_amplitude.unsqueeze(0).unsqueeze(0)
        target_binary_batch = target_binary.unsqueeze(0).unsqueeze(0)
        
        final_inefficiency = calculate_inefficiency(final_predicted_amplitude_batch, target_binary_batch)
        final_non_uniformity = calculate_non_uniformity(final_predicted_amplitude_batch, target_binary_batch)
        final_efficiency = 1.0 - final_inefficiency.item()
        
        # Calculate intensity error
        target_intensity_norm = _normalize_intensity((target_binary**2).unsqueeze(0).unsqueeze(0)).squeeze()
        recon_intensity_norm = _normalize_intensity((final_predicted_amplitude**2).unsqueeze(0).unsqueeze(0)).squeeze()
        intensity_error = torch.sqrt(((recon_intensity_norm - target_intensity_norm) ** 2).mean()).item()
        
        final_metrics = {
            'inefficiency': final_inefficiency.item(),
            'non_uniformity': final_non_uniformity.item(),
            'efficiency': final_efficiency,
            'intensity_error': intensity_error,
            'reconstruction_time': reconstruction_time
        }
        
        if verbose:
            print(f"\nFinal Metrics:")
            print(f"Efficiency: {final_efficiency:.4f}")
            print(f"Inefficiency: {final_inefficiency.item():.4f}")
            print(f"Non-uniformity: {final_non_uniformity.item():.4f}")
            print(f"Intensity error: {intensity_error:.4f}")

    # Create visualization if requested
    if visualize:
        visualize_gd_results_with_metrics(
            target_binary, learnable_w, learnable_theta, 
            save_path=save_path, show_metrics=show_metrics, 
            final_loss=loss.item()
        )

    # Return results
    results = {
        'w_i': learnable_w.detach().clone(),
        'theta_i': learnable_theta.detach().clone(),
        'reconstructed_image': final_predicted_amplitude.detach().cpu().numpy(),
        'phase_mask': final_theta_grid.detach().cpu().numpy(),
        'losses': losses,
        'final_loss': losses[-1],
        'loss_reduction': losses[0] - losses[-1],
        'final_metrics': final_metrics,
        'metrics': final_metrics,
        'target_sample': target_binary,
        'reconstruction_time': reconstruction_time,
        'optimization_params': {
            'iterations': iterations,
            'lr': lr,
            'var_cost': var_cost,
            'momentum': momentum,
            'optimizer': 'Adam'
        }
    }
    
    return results


def visualize_gd_results_with_metrics(
    dataset, 
    device=None, 
    num_samples=5, 
    save_path='images/GD_sample_predictions.png',
    show_metrics=True, 
    debug=False,
    iterations=1000, 
    lr=0.01, 
    momentum=0.9, 
    var_cost=1000,
    verbose=False, 
    show_loss_plot=False
) -> Tuple[List, List]:
    """
    Visualize GD optimization results similar to neural network visualization.
    Optimizes each sample individually using train_gd_optimization.
    
    Args:
        dataset: Dataset to test on
        device: Device to run on (or None for auto-detection)
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        show_metrics: Whether to show metrics
        debug: Enable debug prints
        iterations: Number of GD iterations per sample
        lr: Learning rate for GD
        var_cost: Weight for non-uniformity loss
        verbose: Print optimization progress
    
    Returns:
        all_metrics: List of metrics for each sample
        all_results: List of detailed results for each sample
    """
    
    # Device selection
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axs = np.array([axs])
    
    all_metrics = []
    all_results = []
    
    for i, idx in enumerate(indices):
        if verbose or debug:
            print(f"\n--- Processing Sample {i+1}/{num_samples} (Index {idx}) ---")
        
        # Get target sample
        target_sample = dataset[idx].to(device)  # Shape: (1, H, W)
        target_binary = target_sample.squeeze(0)  # Shape: (H, W)
        n_points = int(target_binary.sum().item())
        
        if debug:
            print(f"Target shape: {target_binary.shape}")
            print(f"Number of points: {n_points}")
        
        # Run GD optimization on this sample
        gd_result = train_gd_optimization(
            target_sample=target_sample,
            iterations=iterations,
            lr=lr,
            momentum=momentum,
            var_cost=var_cost,
            visualize=False,  # Don't create individual plots
            show_metrics=False,  # We'll calculate metrics ourselves
            verbose=False,  # Suppress individual optimization output
            device=device,
            show_loss_plot=show_loss_plot
        )
        
        # Extract optimized parameters
        w_i_opt = gd_result['w_i']
        theta_i_opt = gd_result['theta_i']
        final_loss = gd_result['final_loss']
        
        # Reconstruct using optimized parameters
        with torch.no_grad():
            # Use the same reconstruction as in train_gd_optimization
            w_grid, theta_grid = w_theta_grid_2d(w_i_opt, theta_i_opt, target_binary)
            complex_grid = w_grid * torch.exp(1j * theta_grid)
            
            # Use angle-free reconstruction (same as training)
            spatial = torch.fft.ifft2(ifftshift(complex_grid, dim=(-2, -1)))
            mag = torch.sqrt((spatial.real**2 + spatial.imag**2) + 1e-12)
            learnable_array = spatial / mag
            
            predicted_ft = fftshift(torch.fft.fft2(learnable_array), dim=(-2, -1))
            predicted_intensity = (predicted_ft.conj() * predicted_ft).real
            predicted_amplitude = torch.sqrt(predicted_intensity + 1e-12)
        
        # Convert to numpy for visualization
        target_np = target_binary.cpu().numpy()
        predicted_amplitude_np = predicted_amplitude.cpu().numpy()
        theta_grid_np = theta_grid.cpu().numpy()
        
        # Calculate normalized intensities for plotting
        target_intensity_norm = _normalize_intensity(
            (target_binary**2).unsqueeze(0).unsqueeze(0)
        ).squeeze().cpu().numpy()
        
        recon_intensity_norm = _normalize_intensity(
            (predicted_amplitude**2).unsqueeze(0).unsqueeze(0)
        ).squeeze().cpu().numpy()
        
        diff_np = recon_intensity_norm - target_intensity_norm
        
        # Calculate metrics using the same approach as train_gd_optimization
        if show_metrics:
            predicted_amplitude_batch = predicted_amplitude.unsqueeze(0).unsqueeze(0)
            target_binary_batch = target_binary.unsqueeze(0).unsqueeze(0)
            
            metrics = calculate_holography_metrics(predicted_amplitude_batch, target_binary_batch)
            inefficiency = metrics['inefficiency'].item()
            non_uniformity = metrics['non_uniformity'].item()
            intensity_error = metrics['intensity_error'].item()
            efficiency = 1.0 - inefficiency
            
            if debug:
                print(f"Efficiency: {efficiency:.4f}")
                print(f"Inefficiency: {inefficiency:.4f}")
                print(f"Non-uniformity: {non_uniformity:.4f}")
                print(f"Intensity error: {intensity_error:.4f}")
            
            all_metrics.append({
                'sample': f'sample_{i}',
                'index': idx,
                'points': n_points,
                'efficiency': efficiency,
                'inefficiency': inefficiency,
                'non_uniformity': non_uniformity,
                'intensity_error': intensity_error,
                'final_loss': final_loss
            })
            
            all_results.append({
                'w_i': w_i_opt.cpu(),
                'theta_i': theta_i_opt.cpu(),
                'target': target_np,
                'reconstructed': predicted_amplitude_np,
                'phase': theta_grid_np,
                'metrics': {
                    'efficiency': efficiency,
                    'inefficiency': inefficiency,
                    'non_uniformity': non_uniformity,
                    'intensity_error': intensity_error,
                    'final_loss': final_loss
                }
            })
        else:
            inefficiency = non_uniformity = intensity_error = 0.0
        
        # Create plots (4 columns per sample)
        # Plot 1: Input intensities |B|²
        im0 = axs[i, 0].imshow(target_intensity_norm, cmap='gray')
        axs[i, 0].set_title(f"Input intensities |B|²\nPoints: {n_points}")
        axs[i, 0].axis('off')
        plt.colorbar(im0, ax=axs[i, 0])
        
        # Plot 2: Predicted phase θ
        im1 = axs[i, 1].imshow(theta_grid_np, cmap='hsv')
        axs[i, 1].set_title("Predicted θ (Phase)")
        axs[i, 1].axis('off')
        plt.colorbar(im1, ax=axs[i, 1])
        
        # Plot 3: Reconstructed intensities |B_recon|²
        im2 = axs[i, 2].imshow(recon_intensity_norm, cmap='gray')
        axs[i, 2].set_title("|B_recon|²")
        axs[i, 2].axis('off')
        plt.colorbar(im2, ax=axs[i, 2])
        
        # Plot 4: Difference with metrics
        im3 = axs[i, 3].imshow(diff_np, cmap='hot')
        if show_metrics:
            axs[i, 3].set_title(f"Difference |B|² - |B_recon|²\n"
                               f"Ineff: {inefficiency:.3f}\n"
                               f"Non-unif: {non_uniformity:.3f}\n"
                               f"Intensity err: {intensity_error:.3f}")
        else:
            axs[i, 3].set_title("Difference |B|² - |B_recon|²")
        axs[i, 3].axis('off')
        plt.colorbar(im3, ax=axs[i, 3])

    plt.suptitle('GD Optimization Results', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    # Print summary table if metrics are enabled
    if show_metrics and all_metrics:
        print("\n" + "="*80)
        print("GD OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        print(f"{'Sample':<12} {'Index':<6} {'Points':<6} {'Efficiency':<10} {'Ineff':<8} {'Non-unif':<8} {'Amp err':<8} {'Final Loss':<10}")
        print("-" * 80)
        for m in all_metrics:
            print(f"{m['sample']:<12} {m['index']:<6} {m['points']:<6} "
                  f"{m['efficiency']:<10.4f} {m['inefficiency']:<8.3f} {m['non_uniformity']:<8.3f} "
                  f"{m['intensity_error']:<8.3f} {m['final_loss']:<10.4f}")
        
        # Calculate averages
        avg_efficiency = np.mean([m['efficiency'] for m in all_metrics])
        avg_ineff = np.mean([m['inefficiency'] for m in all_metrics])
        avg_non_unif = np.mean([m['non_uniformity'] for m in all_metrics])
        avg_intensity_err = np.mean([m['intensity_error'] for m in all_metrics])
        avg_loss = np.mean([m['final_loss'] for m in all_metrics])
        
        print("-" * 80)
        print(f"{'AVERAGE':<12} {'':<6} {'':<6} {avg_efficiency:<10.4f} {avg_ineff:<8.3f} "
              f"{avg_non_unif:<8.3f} {avg_intensity_err:<8.3f} {avg_loss:<10.4f}")
        print("="*80)
    
    return all_metrics, all_results