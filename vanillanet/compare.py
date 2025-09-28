import numpy as np
import torch
import matplotlib.pyplot as plt
from time import perf_counter
from typing import Dict, List, Tuple, Optional

from .gs import gerchberg_saxton_with_metrics
from .utils import w_theta_grid, calculate_complex_ft, inverse_complex_ft
from .metrics import calculate_holography_metrics, get_intensity_normalised_from_amplitude

# Add to compare.py
__all__ = [
    'compare_all_methods',
    'visualize_comparison', 
    'print_comparison_summary'
]

def compare_all_methods(dataset, model, device, num_samples=5, save_path='images/comparison_results.png'):
    """
    Compare Gerchberg-Saxton (standard), Gerchberg-Saxton (weighted), SGD, and Neural Network on the same dataset samples.
    UPDATED: Uses consistent intensity normalization and proper device handling.
    
    Args:
        dataset: Dataset to test on
        model: Trained neural network model
        device: Device to run on (or None for auto-detection)
        num_samples: Number of samples to compare
        save_path: Path to save the comparison plot
    
    Returns:
        dict: Results from all four methods
    """

    from .grad_des import train_gd_optimization

    # Device handling: ensure consistency
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                             'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure model and device are consistent
    model_device = next(model.parameters()).device
    if str(model_device) != str(device):
        print(f"⚠️  Device mismatch detected!")
        print(f"   Model is on: {model_device}")
        print(f"   Requested device: {device}")
        print(f"   Moving model to {device}...")
        model = model.to(device)
    
    print(f"Comparing all methods on {num_samples} samples...")
    print(f"Using device: {device}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Select the same samples for all methods
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Initialize results storage
    all_results = {
        'gs_standard_results': [],
        'gs_weighted_results': [],
        'sgd_results': [],
        'nn_results': [],
        'indices': indices
    }
    
    # Run all four methods on the same samples
    for i, idx in enumerate(indices):
        print(f"\n--- Sample {i+1} (Index {idx}) ---")
        
        # Get target sample - ENSURE CONSISTENT DEVICE HANDLING
        target_raw = dataset[idx]  # Raw tensor from dataset
        target = target_raw.squeeze().numpy()  # For numpy operations
        target_torch = target_raw.unsqueeze(0).to(device)  # For neural network - MOVED TO CORRECT DEVICE
        
        # 1. Gerchberg-Saxton (Standard)
        print("Running Gerchberg-Saxton (Standard)...")
        gs_standard_result = gerchberg_saxton_with_metrics(
            target_torch.squeeze(), 
            num_iterations=2000, 
            convergence_threshold=1e-9,
            device=device, 
            weighted=False
        )
        all_results['gs_standard_results'].append(gs_standard_result)
        
        # 2. Gerchberg-Saxton (Weighted)
        print("Running Gerchberg-Saxton (Weighted)...")
        gs_weighted_result = gerchberg_saxton_with_metrics(
            target_torch.squeeze(), 
            num_iterations=2000, 
            convergence_threshold=1e-9,
            device=device, 
            weighted=True
        )
        all_results['gs_weighted_results'].append(gs_weighted_result)
        
        # 3. SGD optimization
        print("Running GD optimization...")
        sgd_result = train_gd_optimization(
            target_sample=target_torch.squeeze(0), 
            max_points=15, 
            iterations=2000, 
            lr=0.1,
            var_cost=1000,
            visualize=False, 
            show_metrics=True,
            verbose=False,
            device=device
        )
        all_results['sgd_results'].append(sgd_result)
        
        # 4. Neural Network - FIXED DEVICE HANDLING
        print("Running Neural Network...")

        # Time the neural network inference and reconstruction
        start_time = perf_counter()
        with torch.no_grad():
            # ENSURE ALL TENSORS ARE ON THE SAME DEVICE AS MODEL
            target_torch_model = target_torch.to(device)  # Explicitly ensure correct device
            
            # Model inference - now both model and input are on same device
            model_out = model(target_torch_model)
            w_i = model_out[:, :, 0]
            theta_i = model_out[:, :, 1]
            
            # Ensure w_theta_grid gets tensors on correct device
            w_grid, theta_grid = w_theta_grid(w_i, theta_i, target_torch_model)
            
            # Rest of reconstruction pipeline
            _, phi_recon = inverse_complex_ft(w_grid, theta_grid)
            A_const = torch.ones_like(target_torch_model)
            recon, _ = calculate_complex_ft(A_const, phi_recon)
        end_time = perf_counter()
        reconstruction_time = end_time - start_time

        # Calculate metrics using consistent intensity normalization
        metrics = calculate_holography_metrics(recon, target_torch_model)

        nn_result = {
            'phase_mask': theta_grid.squeeze().cpu().numpy(),
            'reconstructed_image': recon.squeeze().cpu().numpy(),
            'target_image': target,
            'metrics': {
                'inefficiency': metrics['inefficiency'].item(),
                'non_uniformity': metrics['non_uniformity'].item(),
                'intensity_error': metrics['intensity_error'].item(),
                'reconstruction_time': reconstruction_time
            }
        }
        all_results['nn_results'].append(nn_result)
    
    # Create comprehensive visualization
    visualize_comparison(all_results, save_path)
    
    # Print comparison summary
    print_comparison_summary(all_results)
    
    return all_results


def visualize_comparison(all_results, save_path='images/comparison_results.png'):
    """
    Visualize comparison between all four methods.
    UPDATED: Uses consistent intensity normalization (sum of intensities = 1).

    Args:
        all_results: Results from compare_all_methods
        save_path: Path to save the plot
    """
    num_samples = len(all_results['gs_standard_results'])
    fig, axs = plt.subplots(num_samples, 16, figsize=(64, 4 * num_samples))  # 4 methods × 4 plots = 16 columns

    # Fix for single sample case - ensure axs is always 2D
    if num_samples == 1:
        axs = axs.reshape(1, -1)

    for i in range(num_samples):
        gs_standard_result = all_results['gs_standard_results'][i]
        gs_weighted_result = all_results['gs_weighted_results'][i]
        sgd_result = all_results['sgd_results'][i]
        nn_result = all_results['nn_results'][i]
        
        # Get common data
        target = gs_standard_result['target_image']  # All use original target
        if hasattr(target, 'cpu'):
            target = target.squeeze().cpu().numpy()
        n_points = int(np.sum(target > 0))
        
        # Calculate target intensity using consistent normalization
        target_torch = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
        target_intensity_norm = get_intensity_normalised_from_amplitude(target_torch).squeeze().numpy()
        
        # Process each method
        methods = [
            ('GS-Std', gs_standard_result),
            ('GS-Wgt', gs_weighted_result),
            ('SGD', sgd_result),
            ('NN', nn_result)
        ]
        
        for method_idx, (method_name, result) in enumerate(methods):
            col_start = method_idx * 4

            # Get reconstructed image and phase
            recon = result['reconstructed_image']
            phase = result['phase_mask']
            
            # Convert to numpy if needed
            if hasattr(recon, 'cpu'):
                recon = recon.squeeze().cpu().numpy()
            if hasattr(phase, 'cpu'):
                phase = phase.squeeze().cpu().numpy()
            
            # Calculate reconstructed intensity using consistent normalization
            recon_torch = torch.from_numpy(recon).float().unsqueeze(0).unsqueeze(0)
            recon_intensity_norm = get_intensity_normalised_from_amplitude(recon_torch).squeeze().numpy()
            
            # Calculate difference
            diff_np = target_intensity_norm - recon_intensity_norm
            
            # Plot 1: Target (only for first method to avoid repetition)
            if method_idx == 0:
                im0 = axs[i, 0].imshow(target_intensity_norm, cmap='gray')
                axs[i, 0].set_title(f"Target |B|²\nPoints: {n_points}")
                axs[i, 0].axis('off')
                plt.colorbar(im0, ax=axs[i, 0])
            else:
                axs[i, col_start].axis('off')
            
            # Plot 2: Phase
            im1 = axs[i, col_start + 1].imshow(phase, cmap='hsv')
            axs[i, col_start + 1].set_title(f"{method_name} Phase")
            axs[i, col_start + 1].axis('off')
            plt.colorbar(im1, ax=axs[i, col_start + 1])
            
            # Plot 3: Reconstructed
            im2 = axs[i, col_start + 2].imshow(recon_intensity_norm, cmap='gray')
            axs[i, col_start + 2].set_title(f"{method_name} |B_recon|²")
            axs[i, col_start + 2].axis('off')
            plt.colorbar(im2, ax=axs[i, col_start + 2])
            
            # Plot 4: Difference with metrics
            im3 = axs[i, col_start + 3].imshow(diff_np, cmap='hot')
            metrics = result['metrics']
            axs[i, col_start + 3].set_title(f"{method_name} Difference\n"
                                            f"Ineff: {metrics['inefficiency']:.3f}\n"
                                            f"Non-unif: {metrics['non_uniformity']:.3f}\n"
                                            f"Intensity err: {metrics['intensity_error']:.3f}")
            axs[i, col_start + 3].axis('off')
            plt.colorbar(im3, ax=axs[i, col_start + 3])

    plt.suptitle('4-Way Method Comparison: GS-Standard vs GS-Weighted vs SGD vs Neural Network', fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def print_comparison_summary(all_results):
    """
    Print a summary comparison of all four methods.
    
    Args:
        all_results: Results from compare_all_methods
    """
    num_samples = len(all_results['gs_standard_results'])
    
    print(f"\n{'='*80}")
    print("4-WAY METHOD COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Calculate average metrics for each method
    methods = ['GS-Standard', 'GS-Weighted', 'SGD', 'NN']
    method_results = [
        all_results['gs_standard_results'], 
        all_results['gs_weighted_results'], 
        all_results['sgd_results'], 
        all_results['nn_results']
    ]
    
    for method_name, results in zip(methods, method_results):
        # All methods now have consistent metrics structure
        avg_ineff = np.mean([r['metrics']['inefficiency'] for r in results])
        avg_non_unif = np.mean([r['metrics']['non_uniformity'] for r in results])
        avg_intensity_err = np.mean([r['metrics']['intensity_error'] for r in results])
        avg_time = np.mean([r['metrics']['reconstruction_time'] for r in results])
        
        print(f"\n{method_name} (Average over {num_samples} samples):")
        print(f"  Inefficiency:    {avg_ineff:.4f}")
        print(f"  Non-uniformity:  {avg_non_unif:.4f}")
        print(f"  Intensity error: {avg_intensity_err:.4f}")
        print(f"  Reconstruction time: {avg_time:.4f} seconds")
    
    # Detailed per-sample comparison
    print(f"\n{'='*80}")
    print("DETAILED PER-SAMPLE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Sample':<8} {'Method':<10} {'Points':<6} {'Ineff':<8} {'Non-unif':<8} {'Intensity err':<12} {'Time (s)':<8}")
    print("-" * 70)
    
    for i in range(num_samples):
        idx = all_results['indices'][i]

        # Get target from GS results (all methods use same target)
        target = all_results['gs_standard_results'][i]['target_image']
        if hasattr(target, 'cpu'):
            target = target.squeeze().cpu().numpy()
        n_points = int(np.sum(target > 0))
        
        for method_name, results in zip(methods, method_results):
            metrics = results[i]['metrics']
            print(f"Sample {i+1:<2} {method_name:<8} {n_points:<6} {metrics['inefficiency']:<8.3f} "
                  f"{metrics['non_uniformity']:<8.3f} {metrics['intensity_error']:<8.3f} {metrics['reconstruction_time']:<8.4f}")