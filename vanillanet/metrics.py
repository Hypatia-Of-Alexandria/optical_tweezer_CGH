import torch
import numpy as np

__all__ = [
    '_normalize_intensity',
    'get_intensity_normalised_from_amplitude',
    'calculate_inefficiency',
    'calculate_non_uniformity', 
    'calculate_amplitude_error',
    'calculate_holography_metrics'
]

def _normalize_intensity(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize intensity to have sum 1
    Args:
        x: torch.Tensor - intensity tensor (batchsize, 1, size, size)
    Returns:
        torch.Tensor - normalized intensity tensor (batchsize, 1, size, size)
    """
    return x / (x.sum(dim=(2, 3), keepdim=True) + 1e-12)

def get_intensity_normalised_from_amplitude(amplitude):
    """
    Normalise the amplitude to have total intensity of 1

    Args:
        amplitude: image amplitude (B, 1, H, W)
    
    Returns:
        intensity_norm: image intensity (B, 1, H, W), normalised to have total intensity of 1
    """
    intensity = amplitude**2
    intensity_norm = intensity / (intensity.sum(dim=(2,3), keepdim=True) + 1e-12)
    return intensity_norm

def calculate_inefficiency(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate inefficiency: fraction of power off-target.
    Args:
        recon: Reconstructed image amplitude (B, 1, H, W)
        target: Target image amplitude (B, 1, H, W) with binary values (0 or 1)
    Returns:
        inefficiency: Fraction of power in reconstruction that's off-target, mean over batch
    """
    intensity_recon = recon ** 2
    intensity_target = target
    intensity_recon_norm = _normalize_intensity(intensity_recon)
    intensity_target_norm = _normalize_intensity(intensity_target)
    intensity_recon_norm_at_target = intensity_recon_norm * target
    intensity_on_target = intensity_recon_norm_at_target.sum(dim=(2, 3))
    total_intensity = intensity_recon_norm.sum(dim=(2, 3))
    inefficiency = 1 - (intensity_on_target / (total_intensity + 1e-12))
    return inefficiency.mean()


def calculate_non_uniformity(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intensity_recon = recon ** 2
    intensity_recon_norm = _normalize_intensity(intensity_recon)
    intensity_recon_norm_at_target = intensity_recon_norm * target
    flat_intensities = intensity_recon_norm_at_target.view(
        intensity_recon_norm_at_target.shape[0],
        intensity_recon_norm_at_target.shape[1],
        -1,
    )
    target_mask_flat = target.view(target.shape[0], target.shape[1], -1)
    non_uniformities = []
    for b in range(flat_intensities.shape[0]):
        sample_intensities = flat_intensities[b].flatten()
        sample_mask = target_mask_flat[b].flatten()
        sample_intensities = sample_intensities[sample_mask > 0]
        if len(sample_intensities) > 1:
            mean_val = sample_intensities.mean()
            std_val = sample_intensities.std()
            non_uniformity = std_val / (mean_val + 1e-12)
            non_uniformities.append(non_uniformity)
        else:
            non_uniformities.append(recon.sum() * 0)  # Preserves gradient chain
    return torch.stack(non_uniformities).mean()


def calculate_intensity_error(recon, target):
    """
    Calculate intensity error: RMS intensity error between target and reconstruction. Error is only calculated in target pixels
    
    Args:
        recon: Reconstructed image amplitude (B, 1, H, W) 
        target: Target image amplitude (B, 1, H, W) 
    
    Returns:
        intensity_error: RMS error in target regions, mean over batch
    """
    
    intensity_recon_norm = get_intensity_normalised_from_amplitude(recon)
    intensity_target_norm = get_intensity_normalised_from_amplitude(target)
    
    # Calculate RMS error only in target regions
    error = (intensity_recon_norm - intensity_target_norm) * target
    rms_error = torch.sqrt((error ** 2).sum(dim=(2,3)))
    
    return rms_error.mean()


def calculate_holography_metrics(recon, target):
    """
    Calculate all holography  at once.
    
    Args:
        recon: Reconstructed image amplitude (B, 1, H, W)
        target: Target image amplitude (B, 1, H, W)
    
    Returns:
        dict: Dictionary containing inefficiency, non_uniformity, intensity_error
    """
    metrics = {}
    metrics['inefficiency'] = calculate_inefficiency(recon, target)
    metrics['non_uniformity'] = calculate_non_uniformity(recon, target)
    metrics['intensity_error'] = calculate_intensity_error(recon, target)
    
    return metrics
