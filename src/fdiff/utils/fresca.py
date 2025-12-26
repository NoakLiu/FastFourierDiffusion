"""FreSca: Frequency Scaling for Diffusion Models.

Based on "FreSca: Scaling in Frequency Space Enhances Diffusion Models"
(arXiv:2504.02154v3). This module provides frequency-based control over
diffusion model outputs by decomposing and scaling low/high frequency components.
"""

from typing import Literal, Optional
import torch
import torch.nn.functional as F


def create_frequency_masks(
    shape: tuple[int, ...],
    cutoff_ratio: float,
    cutoff_strategy: Literal["spatial", "energy"] = "spatial",
    freq_spectrum: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create low-pass and high-pass frequency masks.
    
    Supports both 1D (sequence) and 2D (spatial) frequency domains.
    
    Args:
        shape: Frequency domain dimensions (n_freq,) for 1D or (H, W) for 2D
        cutoff_ratio: Cutoff ratio r0 in [0, 1]
        cutoff_strategy: "spatial" for spatial-ratio cutoff, "energy" for energy-based cutoff
        freq_spectrum: Frequency spectrum for energy-based cutoff (optional)
        
    Returns:
        Tuple of (low_pass_mask, high_pass_mask)
    """
    if len(shape) == 1:
        # 1D case: sequence dimension
        n_freq = shape[0]
        device = freq_spectrum.device if freq_spectrum is not None else torch.device("cpu")
        
        # Frequency indices (distance from DC)
        k = torch.arange(n_freq, device=device).float()
        
        if cutoff_strategy == "spatial":
            # Spatial-ratio cutoff: Rc = r0 * n_freq
            Rc = cutoff_ratio * n_freq
            low_pass_mask = (k <= Rc).float()
        elif cutoff_strategy == "energy":
            if freq_spectrum is None:
                raise ValueError("freq_spectrum required for energy-based cutoff")
            
            # Compute total energy
            Etot = torch.abs(freq_spectrum).sum()
            
            # Find cutoff index
            Rc = 0
            cumulative_energy = 0.0
            for i in range(n_freq):
                cumulative_energy += torch.abs(freq_spectrum[i]).item()
                if cumulative_energy >= cutoff_ratio * Etot.item():
                    Rc = i
                    break
            
            low_pass_mask = (k <= Rc).float()
        else:
            raise ValueError(f"Unknown cutoff_strategy: {cutoff_strategy}")
        
        high_pass_mask = 1.0 - low_pass_mask
        
    elif len(shape) == 2:
        # 2D case: spatial dimensions
        H, W = shape
        device = freq_spectrum.device if freq_spectrum is not None else torch.device("cpu")
        
        # Create frequency coordinate grids
        kx = torch.arange(H, device=device, dtype=torch.float32)
        ky = torch.arange(W, device=device, dtype=torch.float32)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        
        # Distance from DC component
        k_dist = torch.sqrt(kx**2 + ky**2)
        
        if cutoff_strategy == "spatial":
            # Spatial-ratio cutoff: Rc = r0 * min(H/2, W/2)
            Rc = cutoff_ratio * min(H / 2, W / 2)
            low_pass_mask = (k_dist <= Rc).float()
        elif cutoff_strategy == "energy":
            # Energy-based cutoff: find R such that cumulative energy >= r0 * Etot
            if freq_spectrum is None:
                raise ValueError("freq_spectrum required for energy-based cutoff")
            
            # Compute total energy
            Etot = torch.abs(freq_spectrum).sum()
            
            # Find cutoff radius
            Rc = 0
            for R in range(int(min(H, W) / 2) + 1):
                mask = (k_dist <= R).float()
                energy = (torch.abs(freq_spectrum) * mask).sum()
                if energy >= cutoff_ratio * Etot:
                    Rc = R
                    break
            
            low_pass_mask = (k_dist <= Rc).float()
        else:
            raise ValueError(f"Unknown cutoff_strategy: {cutoff_strategy}")
        
        high_pass_mask = 1.0 - low_pass_mask
    else:
        raise ValueError(f"Unsupported shape dimension: {len(shape)}")
    
    return low_pass_mask, high_pass_mask


def frequency_scale(
    x: torch.Tensor,
    low_scale: float = 1.0,
    high_scale: float = 1.0,
    cutoff_ratio: float = 0.5,
    cutoff_strategy: Literal["spatial", "energy"] = "spatial",
    dim: int = 1,
) -> torch.Tensor:
    """Apply frequency scaling to a tensor (FreSca operation).
    
    Decomposes input into low and high frequency components and applies
    independent scaling factors to each.
    
    Args:
        x: Input tensor of shape (batch, seq_len, channels) or (batch, H, W, channels)
        low_scale: Scaling factor for low-frequency components (l)
        high_scale: Scaling factor for high-frequency components (h)
        cutoff_ratio: Cutoff ratio r0 for frequency separation
        cutoff_strategy: "spatial" or "energy" based cutoff
        dim: Dimension along which to apply FFT (default: 1 for sequence dimension)
        
    Returns:
        Scaled tensor with same shape as input
    """
    if low_scale == 1.0 and high_scale == 1.0:
        return x  # No scaling needed
    
    original_shape = x.shape
    device = x.device
    dtype = x.dtype
    
    # Handle different input shapes
    if x.dim() == 3:
        # (batch, seq_len, channels) - apply FFT along sequence dimension
        batch_size, seq_len, n_channels = x.shape
        
        # Apply FFT along sequence dimension for each channel
        x_freq = torch.fft.rfft(x, dim=dim, norm="ortho")  # (batch, n_freq, channels)
        n_freq = x_freq.shape[1]
        
        # Create frequency masks (1D case)
        if cutoff_strategy == "energy":
            # Use magnitude spectrum for energy calculation
            freq_spectrum = torch.abs(x_freq).mean(dim=(0, 2))  # Average over batch and channels: (n_freq,)
            low_mask, high_mask = create_frequency_masks(
                (n_freq,), cutoff_ratio, cutoff_strategy, freq_spectrum
            )
            # Expand masks to match x_freq shape: (n_freq,) -> (1, n_freq, 1)
            low_mask = low_mask.unsqueeze(0).unsqueeze(-1)  # (1, n_freq, 1)
            high_mask = high_mask.unsqueeze(0).unsqueeze(-1)  # (1, n_freq, 1)
        else:
            low_mask, high_mask = create_frequency_masks(
                (n_freq,), cutoff_ratio, cutoff_strategy
            )
            low_mask = low_mask.unsqueeze(0).unsqueeze(-1).to(device)  # (1, n_freq, 1)
            high_mask = high_mask.unsqueeze(0).unsqueeze(-1).to(device)  # (1, n_freq, 1)
        
        # Apply scaling
        x_freq_scaled = (
            low_scale * low_mask * x_freq + high_scale * high_mask * x_freq
        )
        
        # Inverse FFT
        x_scaled = torch.fft.irfft(x_freq_scaled, n=seq_len, dim=dim, norm="ortho")
        
        # Ensure same shape
        if x_scaled.shape[1] != seq_len:
            if x_scaled.shape[1] < seq_len:
                pad_size = seq_len - x_scaled.shape[1]
                x_scaled = F.pad(x_scaled, (0, 0, 0, pad_size), mode='constant', value=0)
            else:
                x_scaled = x_scaled[:, :seq_len, :]
    
    elif x.dim() == 4:
        # (batch, H, W, channels) - apply 2D FFT
        batch_size, H, W, n_channels = x.shape
        
        # Apply 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # (batch, H, n_freq, channels)
        
        # Create frequency masks
        if cutoff_strategy == "energy":
            freq_spectrum = torch.abs(x_freq).mean(dim=(0, 3))  # Average over batch and channels
            low_mask, high_mask = create_frequency_masks(
                (H, x_freq.shape[2]), cutoff_ratio, cutoff_strategy, freq_spectrum
            )
            low_mask = low_mask.unsqueeze(0).unsqueeze(-1)  # (1, H, n_freq, 1)
            high_mask = high_mask.unsqueeze(0).unsqueeze(-1)  # (1, H, n_freq, 1)
        else:
            low_mask, high_mask = create_frequency_masks(
                (H, x_freq.shape[2]), cutoff_ratio, cutoff_strategy
            )
            low_mask = low_mask.unsqueeze(0).unsqueeze(-1).to(device)
            high_mask = high_mask.unsqueeze(0).unsqueeze(-1).to(device)
        
        # Apply scaling
        x_freq_scaled = (
            low_scale * low_mask * x_freq + high_scale * high_mask * x_freq
        )
        
        # Inverse 2D FFT
        x_scaled = torch.fft.irfft2(x_freq_scaled, s=(H, W), dim=(1, 2), norm="ortho")
    
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    
    return x_scaled


def apply_fresca_to_score(
    score: torch.Tensor,
    low_scale: float = 1.0,
    high_scale: float = 1.0,
    cutoff_ratio: float = 0.5,
    cutoff_strategy: Literal["spatial", "energy"] = "energy",
    timestep: Optional[float] = None,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """Apply FreSca frequency scaling to score prediction.
    
    This function implements the core FreSca operation for diffusion models.
    Optionally supports time-dependent scaling schedules.
    
    Args:
        score: Score prediction tensor (batch, seq_len, channels)
        low_scale: Low-frequency scaling factor (l)
        high_scale: High-frequency scaling factor (h)
        cutoff_ratio: Cutoff ratio r0
        cutoff_strategy: "spatial" or "energy" based cutoff
        timestep: Current timestep (for dynamic scheduling)
        num_steps: Total number of diffusion steps (for dynamic scheduling)
        
    Returns:
        Frequency-scaled score tensor
    """
    # Optional: Apply time-dependent scaling schedule
    if timestep is not None and num_steps is not None:
        # Linear decay schedule for high-frequency scaling
        # h(t) = (T-t)/T * (h_max - 1) + 1
        t_normalized = timestep / num_steps if num_steps > 0 else 0.0
        if high_scale > 1.0:
            # Decay high-frequency scaling in early steps
            high_scale_dynamic = (1.0 - t_normalized) * (high_scale - 1.0) + 1.0
        else:
            high_scale_dynamic = high_scale
    else:
        high_scale_dynamic = high_scale
    
    # Apply frequency scaling
    score_scaled = frequency_scale(
        score,
        low_scale=low_scale,
        high_scale=high_scale_dynamic,
        cutoff_ratio=cutoff_ratio,
        cutoff_strategy=cutoff_strategy,
    )
    
    return score_scaled


def analyze_frequency_content(
    x: torch.Tensor,
    cutoff_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Analyze frequency content of a tensor.
    
    Useful for understanding which frequency bands are most active,
    which can inform caching strategies.
    
    Args:
        x: Input tensor (batch, seq_len, channels)
        cutoff_ratio: Cutoff ratio for frequency separation
        
    Returns:
        Dictionary with frequency analysis metrics
    """
    # Apply FFT
    x_freq = torch.fft.rfft(x, dim=1, norm="ortho")
    n_freq = x_freq.shape[1]
    
    # Create masks
    low_mask, high_mask = create_frequency_masks(
        (n_freq, 1), cutoff_ratio, "spatial"
    )
    low_mask = low_mask.unsqueeze(0).unsqueeze(-1).to(x.device)
    high_mask = high_mask.unsqueeze(0).unsqueeze(-1).to(x.device)
    
    # Compute energy in each band
    low_energy = (torch.abs(x_freq) * low_mask).sum()
    high_energy = (torch.abs(x_freq) * high_mask).sum()
    total_energy = torch.abs(x_freq).sum()
    
    return {
        "low_energy": low_energy,
        "high_energy": high_energy,
        "total_energy": total_energy,
        "low_energy_ratio": low_energy / (total_energy + 1e-8),
        "high_energy_ratio": high_energy / (total_energy + 1e-8),
    }

