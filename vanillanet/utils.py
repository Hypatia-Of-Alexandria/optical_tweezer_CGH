import torch

__all__ = [
    'fftshift',
    'ifftshift',
    'calculate_complex_ft',
    'inverse_complex_ft',
    'inverse_complex_ft_grid',
    'w_theta_grid',
    'w_theta_grid_2d'
]

def fftshift(x, dim=(-2, -1)): # dim = (-2, -1) does fft shift along the last two axes (equivalent to dim = (0,1) for a 2D array)
    """
    Apply FFT shift to center the zero-frequency component.
    
    Wrapper around torch.fft.fftshift for consistent dimensional handling
    in holographic computations.
    
    Args:
        x (torch.Tensor): Input tensor to shift
        dim (tuple, optional): Dimensions along which to shift. 
            Defaults to (-2, -1) for last two spatial dimensions.
    
    Returns:
        torch.Tensor: Shifted tensor with zero-frequency at center
        
    Note:
        For 2D images, dim=(-2, -1) shifts along height and width dimensions.
    """

    return torch.fft.fftshift(x, dim=dim)

def ifftshift(x, dim=(-2, -1)):
    """
    Apply inverse FFT shift to move zero-frequency from center.
    
    Inverse operation of fftshift, used before inverse FFT operations
    to ensure proper frequency ordering.
    
    Args:
        x (torch.Tensor): Input tensor to shift
        dim (tuple, optional): Dimensions along which to shift.
            Defaults to (-2, -1) for last two spatial dimensions.
    
    Returns:
        torch.Tensor: Shifted tensor with standard FFT frequency ordering
    """

    return torch.fft.ifftshift(x, dim=dim)

def calculate_complex_ft(A, phi):
    """
    Compute forward Fourier transform of complex amplitude field.
    
    Physics: Implements the discrete Fourier transform of a complex field
    A(x,y) * exp(i*φ(x,y)) to obtain the far-field diffraction pattern
    B(kx,ky) * exp(i*θ(kx,ky)) in k-space (Fourier domain).
    
    This represents the light propagation from the SLM plane (x,y) to the
    far-field observation plane (kx,ky) via Fraunhofer diffraction.
    
    Args:
        A (torch.Tensor): Amplitude field in spatial domain, shape (..., H, W)
        phi (torch.Tensor): Phase field in spatial domain, shape (..., H, W)
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - B: Magnitude of diffraction pattern in k-space, shape (..., H, W)
            - theta: Phase of diffraction pattern in k-space, shape (..., H, W)
            
    Example:
        >>> A = torch.ones(1, 1, 64, 64)  # Uniform amplitude
        >>> phi = torch.zeros(1, 1, 64, 64)  # Zero phase
        >>> B, theta = calculate_complex_ft(A, phi)
        >>> print(B.shape)  # torch.Size([1, 1, 64, 64])
    """
    complex_input = A * torch.exp(1j * phi)
    ft = fftshift(torch.fft.fft2(complex_input), dim=(-2, -1))
    B = torch.abs(ft)
    theta = torch.atan2(ft.imag, ft.real)  # Use torch.atan2 for phase
    return B, theta

def inverse_complex_ft(B, theta):
    """
    Given B and theta (defining a complex spectrum B * exp(j*theta)) in k-space,
    compute the inverse Fourier transform to get (A, phi) in x-space.

    Input shape: (batchsize, 1, size, size)
    Output shape: (batchsize, 1, size, size)
    """
    complex_spectrum = B * torch.exp(1j * theta)
    inverse_ft = torch.fft.ifft2(ifftshift(complex_spectrum, dim=(-2, -1))) # do I need to do an fft shift here (I don't think so)?
    A = torch.abs(inverse_ft)
    phi = torch.atan2(inverse_ft.imag, inverse_ft.real)
    return A, phi

def inverse_complex_ft_grid(complex_grid):
    """
    Given a complex grid, compute the inverse Fourier transform to get (A, phi) in x-space.

    Input shape: (batchsize, 1, size, size)
    Output shape: (batchsize, 1, size, size)
    """
    inverse_ft = torch.fft.ifft2(ifftshift(complex_grid, dim=(-2, -1))) # do I need to do an fft shift here (I don't think so)?
    A = torch.abs(inverse_ft)
    phi = torch.atan2(inverse_ft.imag, inverse_ft.real)
    return A, phi

def w_theta_grid(w, theta, batch):
    """
    Arguments:
    w: torch tensor (batch_size, max_points) - model outputs for max_points
    theta: torch tensor (batch_size, max_points) - model outputs for max_points  
    batch: torch tensor (batch_size, 1, size, size) - target images with variable point counts

    Outputs:
    w_grid: torch tensor (batch_size, 1, size, size)
    theta_grid: torch tensor (batch_size, 1, size, size)
    
    """
    max_points = w.shape[1]
    batch_size = batch.shape[0]

    w_grid = torch.zeros_like(batch)
    theta_grid = torch.zeros_like(batch)

    # Process each sample in the batch separately
    for b in range(batch_size):
        # Find actual bright spots in this sample
        coords = torch.nonzero(batch[b, 0], as_tuple=False)  # (num_actual_points, 2)
        num_actual_points = len(coords)
        
        if num_actual_points > 0:
            # Only use the first num_actual_points from model output
            w_actual = w[b, :num_actual_points]  # (num_actual_points,)
            theta_actual = theta[b, :num_actual_points]  # (num_actual_points,)
            
            # Place values at actual bright spot coordinates
            y_coords = coords[:, 0]
            x_coords = coords[:, 1]
            w_grid[b, 0, y_coords, x_coords] = w_actual
            theta_grid[b, 0, y_coords, x_coords] = theta_actual

    return w_grid, theta_grid

def w_theta_grid_2d(w: torch.Tensor, theta: torch.Tensor, target: torch.Tensor):
    """
    2D version for single-sample GD optimization (from Woj_Test_Sparse_ONLY.ipynb)
    Map 1D per-point predictions (w, theta) onto 2D grid at target locations.
    
    Arguments:
    w: torch tensor (1, max_points) - learnable parameters
    theta: torch tensor (1, max_points) - learnable parameters  
    target: torch tensor (size, size) - 2D target pattern

    Outputs:
    w_grid: torch tensor (size, size)
    theta_grid: torch tensor (size, size)
    """
    # Create grids with correct dtype for float values
    w_grid = torch.zeros_like(target, dtype=torch.float32)
    theta_grid = torch.zeros_like(target, dtype=torch.float32)
    coords = torch.nonzero(target, as_tuple=False)
    num_actual_points = len(coords)
    if num_actual_points > 0:
        w_actual = w[0, :num_actual_points]
        theta_actual = theta[0, :num_actual_points]
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        w_grid = w_grid.clone()
        theta_grid = theta_grid.clone()
        w_grid[y_coords, x_coords] = w_actual
        theta_grid[y_coords, x_coords] = theta_actual
    return w_grid, theta_grid


