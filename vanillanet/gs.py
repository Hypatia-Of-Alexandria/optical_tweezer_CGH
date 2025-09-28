import numpy as np
import torch
import matplotlib.pyplot as plt
from time import perf_counter
from typing import Dict, List, Tuple, Optional

from .device import get_device
from .utils import calculate_complex_ft, inverse_complex_ft_grid
from .metrics import calculate_holography_metrics, get_intensity_normalised_from_amplitude
from .debug import debug_real_data_detailed


__all__ = [
    'GerchbergSaxtonSolver',
    'gerchberg_saxton',
    'gerchberg_saxton_with_metrics', 
    'visualize_gs_results_multirow',
    'test_gs_multiple_samples',
    'test_gerchberg_saxton'
]

class GerchbergSaxtonSolver:
    def __init__(self, target_amplitude, initial_phase=None, device='cpu'):
        """
        Initialize the Gerchberg-Saxton solver.
        
        Args:
            target_amplitude: Target amplitude pattern at the image plane (torch tensor)
            initial_phase: Initial phase guess (if None, starts with random phase)
            device: Computing device ('cpu' or 'cuda')
        """
        self.device = device
        self.target_amplitude = target_amplitude.to(device)
        self.size = target_amplitude.shape[-1]  # Assumes square images
        
        # Initialize SLM amplitude (usually uniform)
        self.slm_amplitude = torch.ones_like(target_amplitude).to(device)
        
        # Initialize phase at SLM plane
        if initial_phase is None:
            self.slm_phase = 2 * np.pi * torch.rand_like(target_amplitude).to(device)
        else:
            self.slm_phase = initial_phase.to(device)

    def iterate(self, num_iterations=1000, convergence_threshold=1e-6, weighted=False, epsilon=1e-10,
            inefficiency_threshold=0.3, non_uniformity_threshold=0.1):
        """
        Perform Gerchberg-Saxton iterations.
        
        Args:
            num_iterations: Maximum number of iterations
            convergence_threshold: Stop if error change is below this threshold
            weighted: If True, use weighted Gerchberg-Saxton algorithm
            epsilon: Small constant to prevent division by zero
            
        Returns:
            slm_phase: Optimized phase mask for the SLM
            errors: Dictionary containing error histories and metrics
        """
        errors = []
        inefficiencies = []
        non_uniformities = []
        
        prev_inefficiency = float('inf')
        prev_non_uniformity = float('inf')

        # Quality thresholds for weighted GS
        if weighted:
            inefficiency_threshold = 0.3
            non_uniformity_threshold = 0.1
            prev_g = torch.ones_like(self.target_amplitude)
            print(f"Weighted GS quality requirements:")
            print(f"  Inefficiency must be < {inefficiency_threshold}")
            print(f"  Non-uniformity must be < {non_uniformity_threshold}")
        
        for i in range(num_iterations):
            # Forward propagation: SLM plane -> Image plane
            B_image, theta_image = calculate_complex_ft(self.slm_amplitude, self.slm_phase)
            
            # Apply amplitude constraint at image plane
            if not weighted:
                constrained_image_field = self.target_amplitude * torch.exp(1j * theta_image)
            else:
                ## 1. Weighted GS with numerical stability (with absolute value), proposed by Lukin's paper, Large-scale uniform optical focus array
                ## generation with a phase spatial light modulator - this gives the "classic" weighted GS
                # fixed - issue with mean on targets resolved
                ## https://opg.optica.org/ol/fulltext.cfm?uri=ol-44-12-3178&id=413736

                B_image_stable = torch.clamp(B_image, min=epsilon)
                B_mean_on_targets = (B_image * self.target_amplitude).sum() / self.target_amplitude.sum().clamp_min(1)
                curr_g = (self.target_amplitude * B_mean_on_targets / B_image_stable) * prev_g
                constrained_image_field = curr_g * torch.exp(1j * theta_image)
                prev_g = curr_g

                ## 2. Weighted GS with ratios, proposed by Yang Wu, Jun Wang, Chun Chen, Chan-Juan Liu, Feng-Ming Jin, and Ni Chen
                ## https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-2-1412&id=446366 - this proposes an alternative weighted GS, is supposed to be better? but is not standard weighted GS
                # curr_g = self.target_amplitude * B_image / (B_image + epsilon)
                # constrained_image_field = curr_g * torch.exp(1j * theta_image)
            
            # Backward propagation: Image plane -> SLM plane  
            A_slm, phi_slm = inverse_complex_ft_grid(constrained_image_field)
            
            # Apply amplitude constraint at SLM plane
            self.slm_phase = phi_slm
            
            # Calculate holographic metrics for convergence
            # Ensure proper shape for metrics calculation: (B, 1, H, W)
            if B_image.dim() == 2:  # (H, W)
                B_image_for_metrics = B_image.unsqueeze(0).unsqueeze(0)
                target_for_metrics = self.target_amplitude.unsqueeze(0).unsqueeze(0)
            elif B_image.dim() == 3:  # (1, H, W) or (C, H, W)
                B_image_for_metrics = B_image.unsqueeze(0)
                target_for_metrics = self.target_amplitude.unsqueeze(0)
            else:  # Already 4D
                B_image_for_metrics = B_image
                target_for_metrics = self.target_amplitude
            
            # Calculate holographic metrics
            metrics = calculate_holography_metrics(B_image_for_metrics, target_for_metrics)
            current_inefficiency = metrics['inefficiency'].item()
            current_non_uniformity = metrics['non_uniformity'].item()
            
            # Store metrics
            inefficiencies.append(current_inefficiency)
            non_uniformities.append(current_non_uniformity)
            
            # Calculate traditional MSE error for logging
            mse_error = torch.mean((B_image - self.target_amplitude) ** 2).item()
            errors.append(mse_error)
            
            # Check for convergence
            inefficiency_change = abs(prev_inefficiency - current_inefficiency)
            non_uniformity_change = abs(prev_non_uniformity - current_non_uniformity)
            
            # Convergence criteria depend on whether weighted GS is used
            if weighted:
                # For weighted GS: require both quality thresholds AND convergence stability
                
                stability_met = (inefficiency_change < convergence_threshold and 
                            non_uniformity_change < convergence_threshold and 
                            i > 10)  # Require more iterations for weighted GS
                
                if stability_met:
                    print(f"Weighted GS converged after {i+1} iterations")
                    print(f"Stability achieved:")
                    print(f"  Inefficiency change: {inefficiency_change:.8f} < {convergence_threshold}")
                    print(f"  Non-uniformity change: {non_uniformity_change:.8f} < {convergence_threshold}")
                    break
                elif i % 50 == 0:
                    print(f"Iteration {i+1}: Waiting for stability...")
                    print(f"  Inefficiency change: {inefficiency_change:.8f} (need < {convergence_threshold})")
                    print(f"  Non-uniformity change: {non_uniformity_change:.8f} (need < {convergence_threshold})")

                    
            else:
                # For standard GS: use original convergence criteria
                if (inefficiency_change < convergence_threshold and 
                    non_uniformity_change < convergence_threshold and 
                    i > 10):
                    print(f"Standard GS converged after {i+1} iterations")
                    print(f"Final inefficiency: {current_inefficiency:.6f} (change: {inefficiency_change:.8f})")
                    print(f"Final non-uniformity: {current_non_uniformity:.6f} (change: {non_uniformity_change:.8f})")
                    break
                
            prev_inefficiency = current_inefficiency
            prev_non_uniformity = current_non_uniformity
            
            if i % 10 == 0:
                print(f"Iteration {i+1}/{num_iterations}")
                print(f"  MSE Error: {mse_error:.6f}")
                print(f"  Inefficiency: {current_inefficiency:.6f}")
                print(f"  Non-uniformity: {current_non_uniformity:.6f}")
                    
        
        return self.slm_phase, {
            'mse_errors': errors,
            'inefficiencies': inefficiencies, 
            'non_uniformities': non_uniformities
        }
            
    def get_reconstructed_image(self):
        """
        Get the reconstructed image amplitude using current SLM phase.
        
        Returns:
            reconstructed_amplitude: Amplitude pattern at image plane
        """
        B_image, _ = calculate_complex_ft(self.slm_amplitude, self.slm_phase)
        return B_image


def gerchberg_saxton(target_image, num_iterations=1000, convergence_threshold=1e-6, device='cpu', weighted=False, epsilon=1e-10):
    """
    Convenience function to run Gerchberg-Saxton algorithm.
    
    Args:
        target_image: Target intensity pattern (torch tensor with values 0 or 1)
        num_iterations: Number of iterations to run
        device: Computing device
        epsilon: Small constant to prevent division by zero
    Returns:
        slm_phase: Phase mask for the SLM
        reconstructed: Reconstructed amplitude pattern
        errors: Error evolution during iterations
    """
    # Convert intensity to amplitude (square root)
    target_amplitude = torch.sqrt(target_image.float())
    
    # Initialize solver
    solver = GerchbergSaxtonSolver(target_amplitude, device=device)
    
    # Run iterations
    slm_phase, errors = solver.iterate(num_iterations=num_iterations, convergence_threshold=convergence_threshold, weighted=weighted, epsilon=epsilon)
    
    # Get final reconstruction amplitude
    reconstructed = solver.get_reconstructed_image()
    
    return slm_phase, reconstructed, errors


def gerchberg_saxton_with_metrics(target_image, num_iterations=1000, device='cpu', convergence_threshold=1e-6, weighted=False, epsilon=1e-10):
    """
    Enhanced Gerchberg-Saxton algorithm that returns detailed metrics.
    
    Args:
        target_image: Target intensity pattern (torch tensor with values 0 or 1)
        num_iterations: Number of iterations to run
        device: Computing device
        epsilon: Small constant to prevent division by zero
        
    Returns:
        dict: Contains phase_mask, reconstructed_image, target_image, and metrics
    """
    # Convert intensity to amplitude (square root)
    target_amplitude = torch.sqrt(target_image.float())
    
    # Initialize solver
    solver = GerchbergSaxtonSolver(target_amplitude, device=device)
    
    # Time the GS iterations
    start_time = perf_counter()
    slm_phase, metric_histories = solver.iterate(num_iterations=num_iterations, 
                                            convergence_threshold=convergence_threshold, 
                                            weighted=weighted, 
                                            epsilon=epsilon)
    end_time = perf_counter()
    reconstruction_time = end_time - start_time
    
    # Get final reconstruction (amplitude)
    reconstructed = solver.get_reconstructed_image()
    
    # Calculate holography metrics
    # Ensure proper shape: (B, 1, H, W)
    if target_amplitude.dim() == 2:  # (H, W)
        target_for_metrics = target_amplitude.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif target_amplitude.dim() == 3:  # (1, H, W) or (C, H, W)
        target_for_metrics = target_amplitude.unsqueeze(0)  # (1, C, H, W)
    else:  # Already 4D
        target_for_metrics = target_amplitude

    if reconstructed.dim() == 2:  # (H, W)
        recon_for_metrics = reconstructed.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif reconstructed.dim() == 3:  # (1, H, W) or (C, H, W)
        recon_for_metrics = reconstructed.unsqueeze(0)  # (1, C, H, W)
    else:  # Already 4D
        recon_for_metrics = reconstructed
    
    metrics = calculate_holography_metrics(recon_for_metrics, target_for_metrics)
    
    return {
        'phase_mask': slm_phase,
        'reconstructed_image': reconstructed, #amplitude
        'target_image': target_amplitude,
        'errors': metric_histories['mse_errors'],
        'inefficiencies': metric_histories['inefficiencies'],
        'non_uniformities': metric_histories['non_uniformities'],
        'metrics': {
            'inefficiency': metrics['inefficiency'].item(),
            'non_uniformity': metrics['non_uniformity'].item(),
            'intensity_error': metrics['intensity_error'].item(),
            'reconstruction_time': reconstruction_time
        }
    }


def visualize_gs_results_multirow(gs_results_list, save_path='images/gs_results_multirow.png', debug=False, weighted=False):
    """
    Visualize GS algorithm results in a multi-row format matching the neural network visualization.
    """
    from pathlib import Path
    
    num_samples = len(gs_results_list)
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    for i, gs_results in enumerate(gs_results_list):
        target_amplitude = gs_results['target_image']
        recon_amplitude = gs_results['reconstructed_image']
        phase = gs_results['phase_mask']
        metrics = gs_results['metrics']

        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUGGING GS SAMPLE {i+1}")
            print(f"{'='*60}")
            
            # Convert to torch tensors for debug function
            if hasattr(target_amplitude, 'cpu'):
                target_torch = target_amplitude.unsqueeze(0) if target_amplitude.dim() == 3 else target_amplitude
                recon_torch = recon_amplitude.unsqueeze(0) if recon_amplitude.dim() == 3 else recon_amplitude
            else:
                target_torch = torch.from_numpy(target_amplitude).float().unsqueeze(0).unsqueeze(0)
                recon_torch = torch.from_numpy(recon_amplitude).float().unsqueeze(0).unsqueeze(0)
            
            debug_real_data_detailed(recon_torch, target_torch)
            print(f"{'='*60}")
        
        # Convert to numpy if needed
        if hasattr(target_amplitude, 'cpu'):
            target_amplitude = target_amplitude.squeeze().cpu().numpy()
        if hasattr(recon_amplitude, 'cpu'):
            recon_amplitude = recon_amplitude.squeeze().cpu().numpy()
        if hasattr(phase, 'cpu'):
            phase = phase.squeeze().cpu().numpy()
            
        n_points = np.sum(target_amplitude > 0)

        # Calculate intensities using get_intensity_normalised_from_amplitude
        # Convert numpy arrays to torch tensors with proper shape (1, 1, H, W)
        target_torch = torch.from_numpy(target_amplitude).float().unsqueeze(0).unsqueeze(0)
        recon_torch = torch.from_numpy(recon_amplitude).float().unsqueeze(0).unsqueeze(0)

        # Use the consistent function to get normalized intensities
        target_intensity_norm = get_intensity_normalised_from_amplitude(target_torch).squeeze().numpy()
        recon_intensity_norm = get_intensity_normalised_from_amplitude(recon_torch).squeeze().numpy()

        # Calculate difference
        diff_intensity = target_intensity_norm - recon_intensity_norm
        
        # Plot target intensity
        im0 = axs[i, 0].imshow(target_intensity_norm, cmap='gray')
        axs[i, 0].set_title(f"Input |B|²\nPoints: {int(n_points)}")
        axs[i, 0].axis('off')
        plt.colorbar(im0, ax=axs[i, 0])
        
        # Plot phase
        im1 = axs[i, 1].imshow(phase, cmap='hsv')
        axs[i, 1].set_title("Phase φ")
        axs[i, 1].axis('off')
        plt.colorbar(im1, ax=axs[i, 1])
        
        # Plot reconstructed intensity
        im2 = axs[i, 2].imshow(recon_intensity_norm, cmap='gray')
        axs[i, 2].set_title("Recon |B|²")
        axs[i, 2].axis('off')
        plt.colorbar(im2, ax=axs[i, 2])
        
        # Plot difference with metrics
        im3 = axs[i, 3].imshow(diff_intensity, cmap='hot')
        axs[i, 3].set_title(f"Difference\nIneff: {metrics['inefficiency']:.5f}\nNon-unif: {metrics['non_uniformity']:.5f}\nInt err: {metrics['intensity_error']:.5f}")
        axs[i, 3].axis('off')
        plt.colorbar(im3, ax=axs[i, 3])

    plt.suptitle(f'Gerchberg-Saxton Results (weighted: {weighted})', fontsize=16, y=0.98)
    plt.tight_layout(pad=2.0)  # Add padding


    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def test_gs_multiple_samples(dataset, num_samples=5, device=None, debug=False, weighted=False, convergence_threshold=1e-6, epsilon=1e-10, num_iterations=1000):
    """
    Test GS algorithm on multiple samples from dataset with multi-row visualization.
    
    Args:
        dataset: The dataset object to sample from
        num_samples: Number of samples to test (default: 5 for multi-row format)
        device: Device to use for torch operations (optional)
        debug: If True, shows detailed debug information for each sample
    
    Returns:
        List of all GS results
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing GS algorithm on {num_samples} samples...")
    print(f"Using device: {device}")
    
    all_results = []
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1} ---")
        
        # Get a single sample from dataset
        target = dataset[i]  # Shape: (1, 64, 64)
        
        # Run GS algorithm with metrics
        results = gerchberg_saxton_with_metrics(target, num_iterations=num_iterations, device=device, weighted=weighted, convergence_threshold=convergence_threshold, epsilon=epsilon)
        all_results.append(results)
        
        # Print progress
        metrics = results['metrics']
        print(f"Inefficiency: {metrics['inefficiency']:.4f}")
        print(f"Non-uniformity: {metrics['non_uniformity']:.4f}")
        print(f"Intensity error: {metrics['intensity_error']:.4f}")
    
    # Create the multi-row visualization
    print(f"\nCreating {num_samples}-row visualization...")
    visualize_gs_results_multirow(all_results, f'images/gs_results_{num_samples}rowWeighted{weighted}.png', debug=debug, weighted=weighted)
    
    # Print summary table
    print(f"\nGS Algorithm Results Summary:")
    print(f"Weighted: {weighted}")
    print(f"{'Sample':<8} {'Points':<6} {'Ineff':<8} {'Non-unif':<8} {'Intensity err':<8}")
    print("-" * 50)
    for i, results in enumerate(all_results):
        metrics = results['metrics']
        target_amplitude = results['target_image']
        if hasattr(target_amplitude, 'cpu'):
            target_amplitude = target_amplitude.cpu().numpy()
        n_points = int(np.sum(target_amplitude > 0))
        print(f"Sample {i+1:<2} {n_points:<6} {metrics['inefficiency']:<8.3f} "
              f"{metrics['non_uniformity']:<8.3f} {metrics['intensity_error']:<8.3f}")
    
    return all_results


def test_gerchberg_saxton(debug=False, weighted=False, epsilon=1e-10, convergence_threshold=1e-6, num_iterations=1000, num_samples=5):
    """Test the Gerchberg-Saxton implementation with enhanced visualization
        Args:
            debug: If True, shows detailed debug information
            weighted: If True, use weighted Gerchberg-Saxton algorithm
    """
    from .data import DiscretePointsDataset
    
    # Create dataset
    dataset = DiscretePointsDataset(num_samples=num_samples, size=64, min_points=10, max_points=10, 
                                   min_distance=5, border=4)
    
    # Test multiple samples with the new visualization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = test_gs_multiple_samples(dataset, num_samples=num_samples, device=device, debug=debug, weighted=weighted, epsilon=epsilon, convergence_threshold=convergence_threshold, num_iterations=num_iterations)
    
    return all_results
