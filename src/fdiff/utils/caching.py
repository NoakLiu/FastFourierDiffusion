"""E2-CRF: Error-Feedback Event-Driven Cumulative Residual Feature Caching.

This module implements the E2-CRF caching mechanism for accelerating frequency domain
diffusion models as described in the paper.
"""

from typing import Optional

import torch
import torch.nn as nn


class E2CRFCache:
    """Error-Feedback Event-Driven Cumulative Residual Feature Cache.
    
    Implements KV caching for transformer layers with event-driven triggers
    and error-feedback correction to accelerate frequency domain diffusion.
    """

    def __init__(
        self,
        max_len: int,
        num_layers: int,
        d_model: int,
        n_head: int,
        device: torch.device,
        # Caching parameters
        K: int = 5,  # Number of low-frequency tokens always recomputed
        tau_0: float = 0.1,  # Base threshold
        epsilon: float = 1e-6,  # Numerical stability constant
        eta: float = 1e-6,  # Numerical stability for event intensity
        delta_step: int = 1,  # Step interval for event intensity
        R: int = 10,  # Error-feedback correction interval
        tau_warn: float = 0.5,  # Warning threshold for event intensity
        random_probe_ratio: float = 0.05,  # Ratio of high-freq tokens to probe
        alpha_base: float = 0.1,  # Base error-feedback correction strength
    ):
        """Initialize E2-CRF cache.
        
        Args:
            max_len: Maximum sequence length (number of frequency components)
            num_layers: Number of transformer layers
            d_model: Model dimension
            n_head: Number of attention heads
            device: Device to store cache on
            K: Number of low-frequency tokens always recomputed
            tau_0: Base threshold for adaptive caching
            epsilon: Numerical stability constant
            eta: Numerical stability for event intensity
            delta_step: Step interval for computing event intensity
            R: Error-feedback correction interval (every R steps)
            tau_warn: Warning threshold for event intensity
            random_probe_ratio: Ratio of high-frequency tokens to randomly probe
            alpha_base: Base error-feedback correction strength
        """
        self.max_len = max_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.device = device
        
        # Caching parameters
        self.K = K
        self.tau_0 = tau_0
        self.epsilon = epsilon
        self.eta = eta
        self.delta_step = delta_step
        self.R = R
        self.tau_warn = tau_warn
        self.random_probe_ratio = random_probe_ratio
        self.alpha_base = alpha_base
        
        # Cache storage: Cache[layer_idx][token_idx] = (K, V)
        # K, V shape: (batch_size, n_head, head_dim)
        self.kv_cache: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_layers)
        ]
        
        # CRF storage: cumulative residual features at each layer
        # Shape: (num_layers, max_len, d_model) for single sample
        self.crf_cache: Optional[torch.Tensor] = None
        
        # Track previous step for event intensity computation
        self.prev_crf: Optional[torch.Tensor] = None
        self.prev_step: int = -1
        
        # Track current diffusion step
        self.current_step: int = 0
        
        # Statistics
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
            "total_tokens": 0,
        }
    
    def reset(self) -> None:
        """Reset cache for new sampling sequence."""
        self.kv_cache = [{} for _ in range(self.num_layers)]
        self.crf_cache = None
        self.prev_crf = None
        self.prev_step = -1
        self.current_step = 0
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
            "total_tokens": 0,
        }
    
    def compute_event_intensity(
        self, 
        current_crf: torch.Tensor,
        step: int
    ) -> float:
        """Compute CRF residual event intensity.
        
        Args:
            current_crf: Current cumulative residual features (num_layers, max_len, d_model)
            step: Current diffusion step
            
        Returns:
            Event intensity r^(i)
        """
        if self.prev_crf is None or step - self.prev_step < self.delta_step:
            self.prev_crf = current_crf.clone()
            self.prev_step = step
            return 1.0  # High intensity on first step
        
        # Use final layer CRF
        z_L_current = current_crf[-1]  # (max_len, d_model)
        z_L_prev = self.prev_crf[-1]  # (max_len, d_model)
        
        # Compute relative change
        numerator = torch.norm(z_L_current - z_L_prev, dim=-1).pow(2).sum()
        denominator = torch.norm(z_L_prev, dim=-1).pow(2).sum() + self.eta
        
        r = (numerator / denominator).item()
        
        # Update previous
        self.prev_crf = current_crf.clone()
        self.prev_step = step
        
        return r
    
    def determine_recompute_set(
        self,
        x_tilde: torch.Tensor,
        event_intensity: float,
        step: int,
    ) -> set[int]:
        """Determine which tokens to recompute based on event-driven trigger.
        
        Args:
            x_tilde: Frequency domain representation (batch_size, max_len, n_channels)
            event_intensity: Current event intensity
            step: Current diffusion step
            
        Returns:
            Set of token indices to recompute
        """
        # Always recompute low-frequency tokens
        S = set(range(min(self.K, self.max_len)))
        
        # If high event intensity, recompute more tokens
        if event_intensity > self.tau_warn:
            # Recompute all tokens if intensity is very high
            return set(range(self.max_len))
        
        # For cached tokens, check if they need recomputation
        if self.crf_cache is not None and self.prev_crf is not None:
            # Compute token-wise changes
            for k in range(self.K, self.max_len):
                # Compute change in token k
                delta_k = torch.norm(
                    self.crf_cache[-1, k] - self.prev_crf[-1, k],
                    dim=-1
                ).item()
                
                # Energy-weighted threshold
                energy_k = torch.norm(x_tilde[0, k] if x_tilde.dim() == 3 else x_tilde[k]).pow(2).item()
                tau_k = self.tau_0 / (self.epsilon + energy_k)
                
                if delta_k > tau_k:
                    S.add(k)
        
        # Random probe of high-frequency tokens
        high_freq_tokens = list(range(self.K, self.max_len))
        num_probe = max(1, int(len(high_freq_tokens) * self.random_probe_ratio))
        if num_probe > 0 and len(high_freq_tokens) > 0:
            probe_indices = torch.randperm(len(high_freq_tokens), device=self.device)[:num_probe]
            S.update([high_freq_tokens[i] for i in probe_indices.cpu().tolist()])
        
        return S
    
    def get_cached_kv(
        self,
        layer_idx: int,
        token_idx: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV pair for a specific layer and token.
        
        Args:
            layer_idx: Layer index
            token_idx: Token index
            
        Returns:
            Cached (K, V) pair or None if not cached
        """
        if layer_idx < len(self.kv_cache) and token_idx in self.kv_cache[layer_idx]:
            self.stats["cache_hit_count"] += 1
            return self.kv_cache[layer_idx][token_idx]
        return None
    
    def set_cached_kv(
        self,
        layer_idx: int,
        token_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        """Cache KV pair for a specific layer and token.
        
        Args:
            layer_idx: Layer index
            token_idx: Token index
            K: Key tensor
            V: Value tensor
        """
        if layer_idx < len(self.kv_cache):
            self.kv_cache[layer_idx][token_idx] = (K.detach().clone(), V.detach().clone())
            self.stats["recompute_count"] += 1
    
    def update_crf(
        self,
        crf: torch.Tensor,
    ) -> None:
        """Update cumulative residual features cache.
        
        Args:
            crf: Cumulative residual features (num_layers, max_len, d_model)
        """
        self.crf_cache = crf.detach().clone()
    
    def apply_error_feedback(
        self,
        layer_idx: int,
        token_idx: int,
        true_feature: torch.Tensor,
        cached_feature: torch.Tensor,
        event_intensity: float,
    ) -> torch.Tensor:
        """Apply error-feedback correction to cached feature.
        
        Args:
            layer_idx: Layer index
            token_idx: Token index
            true_feature: Ground-truth recomputed feature
            cached_feature: Cached/reused feature
            event_intensity: Current event intensity
            
        Returns:
            Corrected feature
        """
        # Compute error
        error = true_feature - cached_feature
        
        # Adaptive step size based on event intensity
        alpha = self.alpha_base * (1.0 + event_intensity)
        alpha = min(alpha, 1.0)  # Clamp to [0, 1]
        
        # Apply correction
        corrected = cached_feature + alpha * error
        
        return corrected
    
    def should_apply_error_feedback(self, step: int, event_intensity: float) -> bool:
        """Determine if error-feedback correction should be applied.
        
        Args:
            step: Current diffusion step
            event_intensity: Current event intensity
            
        Returns:
            Whether to apply error-feedback correction
        """
        return (step % self.R == 0) or (event_intensity > self.tau_warn)
    
    def get_cache_stats(self) -> dict:
        """Get statistics about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        total_tokens = self.max_len * self.num_layers
        cached_tokens = sum(len(cache) for cache in self.kv_cache)
        cache_ratio = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "cache_ratio": cache_ratio,
            "current_step": self.current_step,
            "recompute_count": self.stats["recompute_count"],
            "cache_hit_count": self.stats["cache_hit_count"],
            "cache_hit_ratio": (
                self.stats["cache_hit_count"] / (self.stats["recompute_count"] + self.stats["cache_hit_count"])
                if (self.stats["recompute_count"] + self.stats["cache_hit_count"]) > 0
                else 0.0
            ),
        }
