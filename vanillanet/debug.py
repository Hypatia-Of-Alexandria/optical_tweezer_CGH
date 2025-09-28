import torch
import numpy as np
from .metrics import get_intensity_normalised_from_amplitude

__all__ = [
    'debug_real_data_detailed'
]

def debug_real_data_detailed(recon, target):
    """
    Detailed debug of your actual reconstructed and target data
    Updated to use current codebase conventions with intensity normalization

    Input: 
    recon: reconstructed image amplitude (B, 1, H, W)
    target: target image amplitude (B, 1, H, W)

    """
    print("=== Real Data Detailed Debug ===")
    
    # Basic statistics
    print(f"Target sum: {target.sum().item()}")
    print(f"Recon amplitude sum: {recon.sum().item()}")
    print(f"Target amplitude non-zero pixels: {(target > 0).sum().item()}")
    print(f"Recon amplitude non-zero pixels: {(recon > 0).sum().item()}")
    print(f"Recon amplitude range: [{recon.min().item():.6f}, {recon.max().item():.6f}]")
    
    # Check amplitude distribution 
    recon_at_target = recon * target
    recon_outside_target = recon * (1 - target)
    
    print(f"\nAmplitude distribution:")
    print(f"Amplitude at target locations: {recon_at_target.sum().item()}")
    print(f"Amplitude outside target: {recon_outside_target.sum().item()}")
    print(f"Fraction amplitude at target (raw): {(recon_at_target.sum() / recon.sum()).item()}")
    
    # Check amplitudes at target locations
    target_amplitudes = recon_at_target[recon_at_target > 0]
    print(f"\nAt target pixel locations (amplitude):")
    print(f"Number of target pixels: {len(target_amplitudes)}")
    print(f"Amplitudes at target: {target_amplitudes.tolist()}")
    
    if len(target_amplitudes) > 0:
        print(f"Mean amplitude: at target {target_amplitudes.mean().item():.6f}")
    else:
        print("Mean amplitude: N/A (no non-zero pixels at target)")
    
    # Check amplitudes outside target locations
    outside_amplitudes = recon_outside_target[recon_outside_target > 0]
    print(f"\nOutside target pixel locations (amplitude):")
    print(f"Number of non-zero pixels: {len(outside_amplitudes)}")
    
    if len(outside_amplitudes) > 0:
        print(f"Mean amplitude: {outside_amplitudes.mean().item():.6f}")
        print(f"Max amplitude: {outside_amplitudes.max().item():.6f}")
    else:
        print("Mean amplitude: N/A (no non-zero pixels outside target)")
        print("Max amplitude: N/A (no non-zero pixels outside target)")
    
    # Now normalize using the current codebase function
    recon_intensity_norm = get_intensity_normalised_from_amplitude(recon) # shape: (B, 1, H, W)
    target_intensity_norm = get_intensity_normalised_from_amplitude(target) # shape: (B, 1, H, W)
    
    print(f"\nAfter intensity normalization (sum of intensities = 1):")
    print(f"Target_intensity_norm sum: {target_intensity_norm.sum().item()}") # due to normalisation, target_intensity_norm.sum().item() = batch_size
    print(f"Recon_intensity_norm sum: {recon_intensity_norm.sum().item()}")
    
    # Calculate inefficiency using normalized intensities
    power_on_target = (recon_intensity_norm * target).sum(dim=(2,3))
    total_power = recon_intensity_norm.sum(dim=(2,3)) # should be 1 due to normalisation
    inefficiency = 1 - (power_on_target / (total_power + 1e-12))
    
    print(f"\nInefficiency calculation (using intensity normalization):")
    print(f"Power on target: {power_on_target.item()}")
    print(f"Total power: {total_power.item()}")
    print(f"Fraction on target: {(power_on_target / total_power).item()}")
    print(f"Inefficiency: {inefficiency.item()}")
    
    # Also show the difference between amplitude and intensity normalization
    print(f"\nComparison with amplitude normalization:")
    recon_amp_norm = recon / (recon.sum(dim=(2,3), keepdim=True) + 1e-12)
    target_amp_norm = target / (target.sum(dim=(2,3), keepdim=True) + 1e-12)
    
    power_on_target_amp = (recon_amp_norm * target).sum(dim=(2,3))
    inefficiency_amp = 1 - (power_on_target_amp / (recon_amp_norm.sum(dim=(2,3)) + 1e-12))
    
    print(f"Amplitude normalization inefficiency: {inefficiency_amp.item()}")
    print(f"Intensity normalization inefficiency: {inefficiency.item()}")
    print(f"Difference: {abs(inefficiency.item() - inefficiency_amp.item()):.6f}")
    
    return inefficiency.item()