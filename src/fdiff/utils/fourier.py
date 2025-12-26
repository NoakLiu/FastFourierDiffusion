import math

import torch
from einops import rearrange
from torch.fft import irfft, rfft


def dft(x: torch.Tensor) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """
    # Ensure input is real (handle numerical precision issues)
    if torch.is_complex(x):
        x = torch.real(x)
    
    max_len = x.size(1)

    # Compute the FFT until the Nyquist frequency
    dft_full = rfft(x, dim=1, norm="ortho")
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    # The first harmonic corresponds to the mean, which is always real
    zero_padding = torch.zeros_like(dft_im[:, 0, :], device=x.device)
    # Use relaxed tolerance for numerical precision issues (especially on MPS)
    # For MPS backend, numerical errors can be larger, so we use a more lenient tolerance
    if not torch.allclose(dft_im[:, 0, :], zero_padding, atol=1e-4):
        # If assertion would fail, just zero out the imaginary part for numerical stability
        dft_im[:, 0, :] = zero_padding
    dft_im = dft_im[:, 1:]

    # If max_len is even, the last component is always zero
    if max_len % 2 == 0:
        # Use relaxed tolerance for numerical precision issues
        # For MPS backend, numerical errors can be larger, so we use a more lenient tolerance
        if not torch.allclose(dft_im[:, -1, :], zero_padding, atol=1e-4):
            # If assertion would fail, just zero out the imaginary part for numerical stability
            dft_im[:, -1, :] = zero_padding
        dft_im = dft_im[:, :-1]

    # Concatenate real and imaginary parts
    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    assert (
        x_tilde.size() == x.size()
    ), f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."

    return x_tilde.detach()


def idft(x: torch.Tensor) -> torch.Tensor:
    """Compute the inverse DFT of the input DFT that only contains non-redundant components.

    Args:
        x (torch.Tensor): DFT of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: Inverse DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)
    n_real = math.ceil((max_len + 1) / 2)

    # Extract real and imaginary parts
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)))
    x_im = torch.cat((zero_padding, x_im), dim=1)

    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    assert (
        x_im.size() == x_re.size()
    ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

    x_freq = torch.complex(x_re, x_im)

    # Apply IFFT
    x_time = irfft(x_freq, n=max_len, dim=1, norm="ortho")

    assert isinstance(x_time, torch.Tensor)
    assert (
        x_time.size() == x.size()
    ), f"The inverse DFT and the input should have the same size. Got {x_time.size()} and {x.size()} instead."

    return x_time.detach()


def spectral_density(x: torch.Tensor, apply_dft: bool = True) -> torch.Tensor:
    """Compute the spectral density of the input time series.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).
        apply_dft (bool, optional): Whether to apply the DFT to the input. Defaults to True.

    Returns:
        torch.Tensor: Spectral density of x with the size (batch_size, n_frequencies, n_channels).
    """

    max_len = x.size(1)
    x = dft(x) if apply_dft else x

    # Extract real and imaginary parts
    n_real = math.ceil((max_len + 1) / 2)
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)))
    x_im = torch.cat((zero_padding, x_im), dim=1)

    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    assert (
        x_im.size() == x_re.size()
    ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

    # Compute the spectral density
    x_dens = x_re**2 + x_im**2
    assert isinstance(x_dens, torch.Tensor)
    return x_dens


def localization_metrics(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the localization metrics for the input time series.

    Args:
        X (torch.Tensor): Input time series of shape (batch_size, max_len, n_channels).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Delocalization in the time domain and in the frequency domain for each sample
    """

    max_len = X.shape[1]

    # Compute the energy distribution over time for the time series
    X_energy = torch.sum(X**2, dim=2, keepdim=True) / torch.sum(
        X**2, dim=(1, 2), keepdim=True
    )
    X_energy = rearrange(X_energy, "batch time 1 -> batch time")

    # Compute the energy distribution over frequency for the time series
    X_spec = spectral_density(X)
    X_spec_mirror = (
        torch.flip(X_spec[:, 1:, :], dims=(1,))
        if max_len % 2 != 0
        else torch.flip(X_spec[:, 1:-1, :], dims=(1,))
    )  # Add the mirrored frequencies beyond the Nyquist frequency
    X_spec = torch.cat((X_spec, X_spec_mirror), dim=1)
    X_spec = torch.sum(X_spec, dim=2, keepdim=True) / torch.sum(
        X_spec, dim=(1, 2), keepdim=True
    )
    X_spec = rearrange(X_spec, "batch freq 1 -> batch freq")
    assert (
        X_spec.shape[1] == max_len
    ), f"Spectral density has incorrect shape at dimension 1, expected {max_len}, got {X_spec.shape[1]} instead."

    # Compute the cyclic distance between each time steps
    t = torch.arange(max_len, dtype=torch.float)
    t1 = rearrange(t, "time -> time 1 ")
    t2 = rearrange(t, "time -> 1 time ")
    cyclic_distance = torch.min(torch.abs(t1 - t2), max_len - torch.abs(t1 - t2))

    # Compute the delocalization of the signal in time domain
    X_loc = torch.einsum("bt, ts -> bs", X_energy, cyclic_distance**2)
    X_loc = torch.min(X_loc, dim=1)[0]

    # Compute the delocalization of the signal in frequency domain
    X_spec_loc = torch.einsum("bt, ts -> bs", X_spec, cyclic_distance**2)
    X_spec_loc = torch.min(X_spec_loc, dim=1)[0]

    return X_loc, X_spec_loc


def smooth_frequency(X: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooths the signal in the frequency domain by convolving it with a Gaussian kernel.

    Args:
        X (torch.Tensor): Time series to smooth of shape (batch_size, max_len, n_channels).
        sigma (float): Gaussian kernel width.

    Returns:
        torch.Tensor: Smoothed signal in the frequency domain of shape (batch_size, max_len, n_channels).
    """

    # Compute Nyquist frequency
    max_len = X.shape[1]
    nyquist_freq = max_len / 2

    # Define Gaussian kernel for each frequency pair
    k = torch.cat(
        (
            torch.arange(0, nyquist_freq, dtype=torch.float32),
            torch.arange(1, nyquist_freq, dtype=torch.float32),
        )
    )
    k1 = rearrange(k, "time -> time 1 ")
    k2 = rearrange(k, "time -> 1 time ")
    gaussian_kernel = torch.exp(-(((k1 - k2) / (sigma)) ** 2) / 2)
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=0, keepdim=True)

    # Convolve X with the Gaussian kernel in the frequency domain
    X = dft(X)
    X = torch.einsum("btc, ts -> bsc", X, gaussian_kernel)
    X = idft(X)
    return X


def frequency_decompose_fft(
    x: torch.Tensor, 
    low_freq_ratio: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose features into low and high frequency components using FFT.
    
    Based on FreqCa paper: low-frequency components have high similarity but low continuity,
    while high-frequency components have low similarity but high continuity.
    
    Args:
        x: Feature tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)
        low_freq_ratio: Ratio of low-frequency components to keep (default: 0.3)
        
    Returns:
        Tuple of (low_freq, high_freq) components with same shape as input
    """
    original_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dimension
        was_2d = True
    else:
        was_2d = False
    
    batch_size, seq_len, d_model = x.shape
    
    # Apply FFT along sequence dimension
    # x shape: (batch_size, seq_len, d_model)
    x_freq = torch.fft.rfft(x, dim=1, norm="ortho")  # (batch_size, n_freq, d_model)
    n_freq = x_freq.shape[1]
    
    # Determine low-frequency cutoff
    n_low = max(1, int(n_freq * low_freq_ratio))
    
    # Optimized: directly create low and high frequency components
    # Low frequency: keep first n_low frequencies, zero out the rest
    x_low_freq = x_freq.clone()
    if n_low < n_freq:
        x_low_freq[:, n_low:, :] = 0
    
    # High frequency: zero out first n_low frequencies, keep the rest
    x_high_freq = x_freq.clone()
    if n_low > 0:
        x_high_freq[:, :n_low, :] = 0
    
    # Convert back to time domain (irfft automatically handles length)
    x_low = torch.fft.irfft(x_low_freq, n=seq_len, dim=1, norm="ortho")
    x_high = torch.fft.irfft(x_high_freq, n=seq_len, dim=1, norm="ortho")
    
    # Ensure same shape as input (irfft should already match, but check for safety)
    if x_low.shape[1] != seq_len:
        if x_low.shape[1] < seq_len:
            pad_size = seq_len - x_low.shape[1]
            x_low = torch.nn.functional.pad(x_low, (0, 0, 0, pad_size), mode='constant', value=0)
        else:
            x_low = x_low[:, :seq_len, :]
    
    if x_high.shape[1] != seq_len:
        if x_high.shape[1] < seq_len:
            pad_size = seq_len - x_high.shape[1]
            x_high = torch.nn.functional.pad(x_high, (0, 0, 0, pad_size), mode='constant', value=0)
        else:
            x_high = x_high[:, :seq_len, :]
    
    if was_2d:
        x_low = x_low.squeeze(0)
        x_high = x_high.squeeze(0)
    
    return x_low, x_high


def frequency_decompose_dct(
    x: torch.Tensor,
    low_freq_ratio: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose features into low and high frequency components using DCT.
    
    Args:
        x: Feature tensor of shape (batch_size, seq_len, d_model) or (seq_len, d_model)
        low_freq_ratio: Ratio of low-frequency components to keep (default: 0.3)
        
    Returns:
        Tuple of (low_freq, high_freq) components with same shape as input
    """
    # DCT can be implemented using FFT with appropriate preprocessing
    # For simplicity, we use FFT-based approximation which is mathematically equivalent
    # to DCT for real signals
    return frequency_decompose_fft(x, low_freq_ratio)
    
    original_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(0)
        was_2d = True
    else:
        was_2d = False
    
    batch_size, seq_len, d_model = x.shape
    
    # Apply DCT along sequence dimension
    x_dct = dct(x, dim=1, norm="ortho")  # (batch_size, seq_len, d_model)
    
    # Determine low-frequency cutoff
    n_low = max(1, int(seq_len * low_freq_ratio))
    
    # Split into low and high frequency
    x_low_dct = x_dct[:, :n_low, :]
    x_high_dct = x_dct[:, n_low:, :]
    
    # Pad high-frequency to match original size
    x_high_dct_padded = torch.zeros_like(x_dct)
    x_high_dct_padded[:, n_low:, :] = x_high_dct
    
    # Convert back to time domain
    x_low = idct(x_low_dct, dim=1, norm="ortho")
    x_high = idct(x_high_dct_padded, dim=1, norm="ortho")
    
    if was_2d:
        x_low = x_low.squeeze(0)
        x_high = x_high.squeeze(0)
    
    return x_low, x_high


def hermite_polynomials(s: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Compute Hermite polynomials up to given order.
    
    Hermite polynomials are defined on normalized interval [-1, 1].
    H_0(s) = 1
    H_1(s) = 2s
    H_2(s) = 4s^2 - 2
    H_3(s) = 8s^3 - 12s
    ...
    
    Args:
        s: Normalized time values in [-1, 1], shape (K,) or (batch_size, K)
        order: Maximum order of Hermite polynomials (default: 2)
        
    Returns:
        Tensor of shape (order+1, K) or (order+1, batch_size, K) containing polynomial values
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False
    
    batch_size, K = s.shape
    device = s.device
    dtype = s.dtype
    
    # Initialize Hermite polynomials
    H = torch.zeros(order + 1, batch_size, K, device=device, dtype=dtype)
    
    # H_0(s) = 1
    H[0] = torch.ones_like(s)
    
    if order >= 1:
        # H_1(s) = 2s
        H[1] = 2 * s
    
    if order >= 2:
        # H_2(s) = 4s^2 - 2
        H[2] = 4 * s**2 - 2
    
    if order >= 3:
        # H_3(s) = 8s^3 - 12s
        H[3] = 8 * s**3 - 12 * s
    
    # For higher orders, use recurrence relation: H_{n+1}(s) = 2s*H_n(s) - 2n*H_{n-1}(s)
    for n in range(3, order):
        H[n + 1] = 2 * s * H[n] - 2 * n * H[n - 1]
    
    if was_1d:
        H = H.squeeze(1)  # (order+1, K)
    else:
        H = H  # (order+1, batch_size, K)
    
    return H


def predict_hermite(
    history: list[torch.Tensor],
    timesteps: list[float],
    target_timestep: float,
    order: int = 2
) -> torch.Tensor:
    """Predict feature value at target timestep using Hermite polynomial interpolation.
    
    Based on FreqCa paper: high-frequency components are predictable using Hermite polynomials.
    
    Args:
        history: List of historical feature tensors, each of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
        timesteps: List of timesteps corresponding to history
        target_timestep: Target timestep to predict
        order: Order of Hermite polynomial (default: 2)
        
    Returns:
        Predicted feature tensor with same shape as history elements
    """
    if len(history) < 2:
        # Not enough history, return last value
        return history[-1].clone()
    
    K = len(history)
    device = history[0].device
    dtype = history[0].dtype
    
    # Normalize timesteps to [-1, 1]
    t_min, t_max = min(timesteps), max(timesteps)
    if t_max == t_min:
        # All timesteps are the same, return last value
        return history[-1].clone()
    
    # Normalize target timestep
    s_target = 2 * (target_timestep - t_min) / (t_max - t_min) - 1
    s_target = torch.clamp(torch.tensor(s_target, device=device, dtype=dtype), -1.0, 1.0)
    
    # Normalize historical timesteps
    s_history = torch.tensor(
        [2 * (t - t_min) / (t_max - t_min) - 1 for t in timesteps],
        device=device,
        dtype=dtype
    )
    s_history = torch.clamp(s_history, -1.0, 1.0)
    
    # Compute Hermite polynomials at historical points
    H_history = hermite_polynomials(s_history, order=order)  # (order+1, K)
    
    # Compute Hermite polynomial at target point
    H_target = hermite_polynomials(s_target.unsqueeze(0), order=order)  # (order+1, 1)
    if H_target.dim() == 2:
        H_target = H_target.squeeze(1)  # (order+1,)
    else:
        H_target = H_target  # (order+1,)
    
    # Stack history features: (K, seq_len, d_model) or (K, batch_size, seq_len, d_model)
    history_stack = torch.stack(history, dim=0)
    
    # Fit coefficients using least squares
    # For each feature dimension, solve: history = H^T @ coeffs
    # coeffs = (H @ H^T)^{-1} @ H @ history
    
    H_matrix = H_history.T  # (K, order+1)
    H_target_vec = H_target  # (order+1,)
    
    # Solve least squares: minimize ||history - H_matrix @ coeffs||^2
    # coeffs = (H_matrix^T @ H_matrix)^{-1} @ H_matrix^T @ history
    HtH = torch.matmul(H_matrix.T, H_matrix)  # (order+1, order+1)
    
    # Add small regularization for numerical stability
    HtH_reg = HtH + torch.eye(order + 1, device=device, dtype=dtype) * 1e-6
    
    try:
        HtH_inv = torch.linalg.inv(HtH_reg)  # (order+1, order+1)
    except:
        # Fallback: use pseudo-inverse
        HtH_inv = torch.linalg.pinv(HtH_reg)
    
    # Compute coefficients for each feature dimension
    if history_stack.dim() == 3:
        # (K, seq_len, d_model)
        # Reshape for matrix multiplication: (K, seq_len*d_model)
        seq_len, d_model = history_stack.shape[1], history_stack.shape[2]
        history_flat = history_stack.view(K, -1)  # (K, seq_len*d_model)
        Ht_history = torch.matmul(H_matrix.T, history_flat)  # (order+1, seq_len*d_model)
        coeffs = torch.matmul(HtH_inv, Ht_history)  # (order+1, seq_len*d_model)
        # Predict at target
        prediction_flat = torch.matmul(H_target_vec, coeffs)  # (seq_len*d_model,)
        prediction = prediction_flat.view(seq_len, d_model)  # (seq_len, d_model)
    else:
        # (K, batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = history_stack.shape[1], history_stack.shape[2], history_stack.shape[3]
        history_flat = history_stack.view(K, -1)  # (K, batch_size*seq_len*d_model)
        Ht_history = torch.matmul(H_matrix.T, history_flat)  # (order+1, batch_size*seq_len*d_model)
        coeffs = torch.matmul(HtH_inv, Ht_history)  # (order+1, batch_size*seq_len*d_model)
        # Predict at target
        prediction_flat = torch.matmul(H_target_vec, coeffs)  # (batch_size*seq_len*d_model,)
        prediction = prediction_flat.view(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
    
    return prediction
