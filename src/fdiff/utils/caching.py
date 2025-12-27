"""E2-CRF: Error-Feedback Event-Driven Cumulative Residual Feature Caching.

This module implements the E2-CRF caching mechanism for accelerating frequency domain
diffusion models as described in the paper.

Enhanced with FreqCa (Frequency-aware Caching) based on:
"FREQCA: ACCELERATING DIFFUSION MODELS VIA FREQUENCY-AWARE CACHING"
"""

from typing import Optional, Literal

import torch
import torch.nn as nn

from fdiff.utils.fourier import frequency_decompose_fft, frequency_decompose_dct, predict_hermite
from fdiff.utils.fresca import analyze_frequency_content


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
        # FreqCa parameters
        use_freqca: bool = True,  # Enable frequency-aware caching
        freq_decomp: Literal["fft", "dct"] = "dct",  # Frequency decomposition method
        low_freq_ratio: float = 0.3,  # Ratio of low-frequency components
        hermite_order: int = 2,  # Order of Hermite polynomial for high-freq prediction
        max_history: int = 3,  # Maximum history length for Hermite prediction
        # FreqCa optimization parameters
        freq_decomp_interval: int = 10,  # Decompose every N steps (default: 10, larger = less frequent)
        freq_decomp_change_threshold: float = 0.01,  # Only decompose if CRF change > threshold
        # FreSca integration parameters
        use_fresca_in_cache: bool = False,  # Use FreSca frequency analysis for cache optimization
        fresca_adaptive_threshold: bool = True,  # Adaptively adjust thresholds based on frequency content
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
            use_freqca: Enable frequency-aware caching
            freq_decomp: Frequency decomposition method ("fft" or "dct")
            low_freq_ratio: Ratio of low-frequency components to keep
            hermite_order: Order of Hermite polynomial for high-frequency prediction
            max_history: Maximum history length for Hermite prediction
            freq_decomp_interval: Interval for frequency decomposition (larger = less frequent, default: 10)
            freq_decomp_change_threshold: Threshold for CRF change to trigger decomposition (default: 0.01)
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
        
        # FreqCa parameters
        self.use_freqca = use_freqca
        self.freq_decomp = freq_decomp
        self.low_freq_ratio = low_freq_ratio
        self.hermite_order = hermite_order
        self.max_history = max_history
        # Optimization: control frequency decomposition frequency
        self.freq_decomp_interval = freq_decomp_interval
        self.freq_decomp_change_threshold = freq_decomp_change_threshold
        self._last_decomp_step: int = -1  # Track last decomposition step
        self._last_crf_for_decomp: Optional[torch.Tensor] = None  # Track last CRF used for decomposition
        
        # FreSca integration
        self.use_fresca_in_cache = use_fresca_in_cache
        self.fresca_adaptive_threshold = fresca_adaptive_threshold
        self._last_freq_analysis: Optional[dict] = None  # Store last frequency analysis
        
        # Cache storage: Cache[layer_idx][token_idx] = (K, V)
        # K, V shape: (batch_size, n_head, head_dim)
        self.kv_cache: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_layers)
        ]
        
        # CRF storage: cumulative residual features at each layer
        # Shape: (num_layers, max_len, d_model) for single sample
        self.crf_cache: Optional[torch.Tensor] = None
        
        # FreqCa: Frequency-decomposed CRF cache
        # Low-frequency: directly reused (high similarity)
        # High-frequency: predicted using Hermite polynomials (high continuity)
        self.crf_low_cache: Optional[torch.Tensor] = None
        self.crf_high_history: list[torch.Tensor] = []  # History for Hermite prediction
        self.crf_timestep_history: list[float] = []  # Corresponding timesteps
        
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
            "freq_decomp_count": 0,  # Track frequency decomposition count
            "freq_decomp_skipped": 0,  # Track skipped decompositions
        }
        
        # Track previous energy for lightweight event intensity computation
        self._prev_energy: Optional[torch.Tensor] = None
    
    def reset(self) -> None:
        """Reset cache for new sampling sequence."""
        self.kv_cache = [{} for _ in range(self.num_layers)]
        self.crf_cache = None
        self.crf_low_cache = None
        self.crf_high_history = []
        self.crf_timestep_history = []
        self.prev_crf = None
        self.prev_step = -1
        self.current_step = 0
        self._last_decomp_step = -1
        self._last_crf_for_decomp = None
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
            "total_tokens": 0,
            "freq_decomp_count": 0,
            "freq_decomp_skipped": 0,
        }
        self._prev_energy = None
    
    def compute_event_intensity(
        self, 
        current_crf: torch.Tensor,
        step: int
    ) -> float:
        """Compute CRF residual event intensity.
        
        Optimized: Use detach() instead of clone() for better performance.
        
        Args:
            current_crf: Current cumulative residual features (num_layers, max_len, d_model)
            step: Current diffusion step
            
        Returns:
            Event intensity r^(i)
        """
        if self.prev_crf is None or step - self.prev_step < self.delta_step:
            # Use detach() instead of clone() for better performance
            if current_crf.device != self.device:
                self.prev_crf = current_crf.detach().to(self.device, non_blocking=True)
            else:
                self.prev_crf = current_crf.detach()
            self.prev_step = step
            # Return lower intensity on first step to allow caching
            # Only return high intensity if this is truly the first step (step == 0)
            return 0.1 if step > 0 else 1.0
        
        # Use final layer CRF
        z_L_current = current_crf[-1]  # (max_len, d_model)
        z_L_prev = self.prev_crf[-1]  # (max_len, d_model)
        
        # Compute relative change (optimized: use in-place operations where possible)
        diff = z_L_current - z_L_prev
        numerator = torch.norm(diff, dim=-1).pow(2).sum()
        denominator = torch.norm(z_L_prev, dim=-1).pow(2).sum() + self.eta
        
        r = (numerator / denominator).item()
        
        # Update previous (use detach() instead of clone())
        if current_crf.device != self.device:
            self.prev_crf = current_crf.detach().to(self.device, non_blocking=True)
        else:
            self.prev_crf = current_crf.detach()
        self.prev_step = step
        
        return r
    
    def determine_recompute_set(
        self,
        x_tilde: torch.Tensor,
        event_intensity: float,
        step: int,
    ) -> set[int]:
        """Determine which tokens to recompute based on event-driven trigger.
        
        MACRO-LEVEL CACHING STRATEGY:
        - Fixed interval recomputation: only recompute every N steps
        - Simple rule: always recompute first K tokens, cache the rest
        - Skip expensive computations (event intensity, delta checks) most of the time
        
        Args:
            x_tilde: Frequency domain representation (batch_size, max_len, n_channels)
            event_intensity: Current event intensity (may be ignored for macro strategy)
            step: Current diffusion step
            
        Returns:
            Set of token indices to recompute
        """
        # MACRO STRATEGY: Fixed interval caching
        # Only recompute at fixed intervals, otherwise use cached values
        recompute_interval = max(1, self.R)  # Use R as the recompute interval
        
        # Check if we should recompute at this step
        should_recompute_all = (step % recompute_interval == 0) or (step == 0)
        
        if should_recompute_all:
            # Full recomputation: always recompute low-frequency tokens
            # For high-frequency tokens, use simple heuristic
            S = set(range(min(self.K, self.max_len)))
            
            # Simple heuristic: if event intensity is very high, recompute more
            if event_intensity > self.tau_warn:
                # Recompute all tokens periodically
                return set(range(self.max_len))
            
            # Otherwise, only recompute low-frequency tokens (K tokens)
            # High-frequency tokens will be cached
            return S
        else:
            # CACHED MODE: Only recompute low-frequency tokens
            # All high-frequency tokens use cache (no expensive checks)
            return set(range(min(self.K, self.max_len)))
    
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
        if layer_idx < len(self.kv_cache):
            layer_cache = self.kv_cache[layer_idx]
            if token_idx in layer_cache:
                self.stats["cache_hit_count"] += 1
                return layer_cache[token_idx]
        return None
    
    def get_cached_kv_batch(
        self,
        layer_idx: int,
        token_indices: list[int],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Batch get cached KV pairs for multiple tokens.
        
        Args:
            layer_idx: Layer index
            token_indices: List of token indices
            
        Returns:
            Dictionary mapping token_idx to (K, V) pair
        """
        result = {}
        if layer_idx < len(self.kv_cache):
            layer_cache = self.kv_cache[layer_idx]
            for token_idx in token_indices:
                if token_idx in layer_cache:
                    self.stats["cache_hit_count"] += 1
                    result[token_idx] = layer_cache[token_idx]
        return result
    
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
            # Only clone and move if necessary (optimize for performance)
            if K.device != self.device:
                # Move to device, detach to break gradient
                K = K.detach().to(self.device, non_blocking=True)
                V = V.detach().to(self.device, non_blocking=True)
            else:
                # Just detach to break gradient, no clone needed if already on device
                K = K.detach()
                V = V.detach()
            self.kv_cache[layer_idx][token_idx] = (K, V)
            self.stats["recompute_count"] += 1
    
    def update_crf(
        self,
        crf: torch.Tensor,
        timestep: Optional[float] = None,
    ) -> None:
        """Update cumulative residual features cache.
        
        Enhanced with FreqCa: decompose CRF into low and high frequency components.
        Low-frequency: directly cached for reuse (high similarity)
        High-frequency: stored in history for Hermite prediction (high continuity)
        
        Optimized: Only perform frequency decomposition when:
        1. Enough steps have passed since last decomposition (interval-based)
        2. CRF has changed significantly (change threshold)
        
        Args:
            crf: Cumulative residual features (num_layers, max_len, d_model)
            timestep: Current diffusion timestep (for Hermite prediction)
        """
        # Store full CRF for backward compatibility
        if crf.device != self.device or crf.requires_grad:
            self.crf_cache = crf.detach().to(self.device, non_blocking=True)
        else:
            self.crf_cache = crf.detach()
        
        # FreqCa: Frequency-aware caching (with optimization)
        if self.use_freqca:
            # Use final layer CRF for frequency decomposition
            crf_final = crf[-1] if crf.dim() == 3 else crf  # (max_len, d_model)
            
            # Optimization: Check if we should perform frequency decomposition
            should_decompose = False
            
            # Check 1: Interval-based (every N steps)
            steps_since_decomp = self.current_step - self._last_decomp_step
            if steps_since_decomp >= self.freq_decomp_interval:
                should_decompose = True
            
            # Check 2: Change-based (only if CRF changed significantly)
            if not should_decompose and self._last_crf_for_decomp is not None:
                # Compute relative change in CRF
                crf_change = torch.norm(crf_final - self._last_crf_for_decomp) / (
                    torch.norm(self._last_crf_for_decomp) + self.epsilon
                )
                if crf_change.item() > self.freq_decomp_change_threshold:
                    should_decompose = True
            elif self._last_crf_for_decomp is None:
                # First time, always decompose
                should_decompose = True
            
            # Perform frequency decomposition if needed
            if should_decompose:
                # Decompose into low and high frequency components
                if self.freq_decomp == "dct":
                    crf_low, crf_high = frequency_decompose_dct(
                        crf_final, low_freq_ratio=self.low_freq_ratio
                    )
                else:  # fft
                    crf_low, crf_high = frequency_decompose_fft(
                        crf_final, low_freq_ratio=self.low_freq_ratio
                    )
                
                # Cache low-frequency component (direct reuse)
                if crf_low.device != self.device or crf_low.requires_grad:
                    self.crf_low_cache = crf_low.detach().to(self.device, non_blocking=True)
                else:
                    self.crf_low_cache = crf_low.detach()
                
                # Update tracking
                self._last_decomp_step = self.current_step
                if crf_final.device != self.device:
                    self._last_crf_for_decomp = crf_final.detach().to(self.device, non_blocking=True)
                else:
                    self._last_crf_for_decomp = crf_final.detach()
                
                self.stats["freq_decomp_count"] += 1
            else:
                self.stats["freq_decomp_skipped"] += 1
            
            # Always store high-frequency in history for Hermite prediction (even if not decomposed)
            # Use cached high-frequency if available, otherwise compute on-the-fly
            if timestep is not None:
                if should_decompose:
                    # We just computed crf_high, use it
                    if crf_high.device != self.device or crf_high.requires_grad:
                        crf_high_cached = crf_high.detach().to(self.device, non_blocking=True)
                    else:
                        crf_high_cached = crf_high.detach()
                else:
                    # Reuse last high-frequency component (approximation)
                    # This is acceptable since high-frequency changes are continuous
                    if len(self.crf_high_history) > 0:
                        crf_high_cached = self.crf_high_history[-1]  # Reuse last
                    else:
                        # Fallback: compute on-the-fly (shouldn't happen often)
                        if self.freq_decomp == "dct":
                            _, crf_high_cached = frequency_decompose_dct(
                                crf_final, low_freq_ratio=self.low_freq_ratio
                            )
                        else:
                            _, crf_high_cached = frequency_decompose_fft(
                                crf_final, low_freq_ratio=self.low_freq_ratio
                            )
                        if crf_high_cached.device != self.device or crf_high_cached.requires_grad:
                            crf_high_cached = crf_high_cached.detach().to(self.device, non_blocking=True)
                        else:
                            crf_high_cached = crf_high_cached.detach()
                
                self.crf_high_history.append(crf_high_cached)
                self.crf_timestep_history.append(timestep)
                
                # Keep only recent history
                if len(self.crf_high_history) > self.max_history:
                    self.crf_high_history.pop(0)
                    self.crf_timestep_history.pop(0)
    
    def predict_crf_freqca(
        self,
        target_timestep: float,
    ) -> Optional[torch.Tensor]:
        """Predict CRF at target timestep using FreqCa strategy.
        
        Low-frequency: directly reused from cache (high similarity)
        High-frequency: predicted using Hermite polynomials (high continuity)
        
        Args:
            target_timestep: Target diffusion timestep
            
        Returns:
            Predicted CRF tensor or None if not enough history
        """
        if not self.use_freqca:
            return None
        
        if self.crf_low_cache is None or len(self.crf_high_history) < 2:
            return None
        
        # Reuse low-frequency component directly
        crf_low_pred = self.crf_low_cache
        
        # Predict high-frequency component using Hermite polynomials
        crf_high_pred = predict_hermite(
            history=self.crf_high_history,
            timesteps=self.crf_timestep_history,
            target_timestep=target_timestep,
            order=self.hermite_order
        )
        
        # Reconstruct CRF: low + high
        crf_pred = crf_low_pred + crf_high_pred
        
        return crf_pred
    
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
        
        stats = {
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
        
        # Add FreqCa statistics if available
        if "freq_decomp_count" in self.stats:
            stats["freq_decomp_count"] = self.stats["freq_decomp_count"]
            stats["freq_decomp_skipped"] = self.stats["freq_decomp_skipped"]
            total_decomp_ops = stats["freq_decomp_count"] + stats["freq_decomp_skipped"]
            if total_decomp_ops > 0:
                stats["freq_decomp_ratio"] = stats["freq_decomp_count"] / total_decomp_ops
            else:
                stats["freq_decomp_ratio"] = 0.0
        
        return stats
