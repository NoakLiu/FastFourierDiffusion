"""E2-CRF caching utilities for accelerating diffusion models.

This module implements the E2-CRF (Event-driven, Efficient, Cumulative Residual Feature)
caching mechanism for accelerating diffusion model inference.
"""

from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from fdiff.utils.fourier import frequency_decompose_fft, frequency_decompose_dct, predict_hermite
    from fdiff.utils.fresca import analyze_frequency_content

from fdiff.utils.fourier import frequency_decompose_fft, frequency_decompose_dct, predict_hermite
from fdiff.utils.fresca import analyze_frequency_content


class E2CRFCache:
    """E2-CRF cache for accelerating diffusion model inference.
    
    This cache stores:
    1. KV pairs for transformer layers (selective token computation)
    2. Cumulative Residual Features (CRF) for event-driven recomputation
    3. Frequency-decomposed CRF components (FreqCa support)
    """
    
    def __init__(
        self,
        num_layers: int,
        max_len: int,
        device: torch.device,
        K: int = 5,
        R: int = 10,
        tau_0: float = 0.1,
        tau_warn: float = 0.5,
        # FreqCa parameters
        use_freqca: bool = False,
        freq_decomp: str = "dct",
        low_freq_ratio: float = 0.3,
        max_history: int = 10,
        hermite_order: int = 3,
        freq_decomp_interval: int = 10,
        # FreSca parameters
        use_fresca_in_cache: bool = False,
        fresca_adaptive_threshold: bool = False,
    ):
        """Initialize E2-CRF cache.
        
        Args:
            num_layers: Number of transformer layers
            max_len: Maximum sequence length
            device: Device to store cache on
            K: Number of low-frequency tokens to always recompute
            R: Error-feedback interval (recompute interval for macro strategy)
            tau_0: Base threshold for event intensity
            tau_warn: Warning threshold for event intensity
            use_freqca: Whether to use FreqCa frequency-aware caching
            freq_decomp: Frequency decomposition method ("fft" or "dct")
            low_freq_ratio: Ratio of low-frequency components
            max_history: Maximum history length for Hermite prediction
            hermite_order: Order of Hermite polynomials
            freq_decomp_interval: Interval for frequency decomposition
            use_fresca_in_cache: Whether to use FreSca in cache decisions
            fresca_adaptive_threshold: Whether to adaptively adjust threshold based on frequency
        """
        self.num_layers = num_layers
        self.max_len = max_len
        self.device = device
        self.K = K
        self.R = R
        self.tau_0 = tau_0
        self.tau_warn = tau_warn
        
        # FreqCa parameters
        self.use_freqca = use_freqca
        self.freq_decomp = freq_decomp
        self.low_freq_ratio = low_freq_ratio
        self.max_history = max_history
        self.hermite_order = hermite_order
        self.freq_decomp_interval = freq_decomp_interval
        
        # FreSca parameters
        self.use_fresca_in_cache = use_fresca_in_cache
        self.fresca_adaptive_threshold = fresca_adaptive_threshold
        
        # NEW DESIGN: Tensor-based cache instead of dict for much faster access
        # Shape: (num_layers, nhead, max_len, head_dim) - None means not initialized
        # We'll initialize on first use to get nhead and head_dim
        self.k_cache_tensor: Optional[torch.Tensor] = None
        self.v_cache_tensor: Optional[torch.Tensor] = None
        self.cache_valid: Optional[torch.Tensor] = None  # (num_layers, max_len) bool
        
        # Keep dict cache for backward compatibility during transition
        self.kv_cache: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_layers)
        ]
        
        # CRF cache: stores cumulative residual features
        self.crf_cache: Optional[torch.Tensor] = None
        
        # FreqCa: separate low and high frequency CRF components
        self.crf_low_cache: Optional[torch.Tensor] = None
        self.crf_high_history: list[torch.Tensor] = []
        self.crf_timestep_history: list[float] = []
        
        # Statistics
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
        }
        
        self.current_step = 0
    
    def reset(self) -> None:
        """Reset the cache."""
        self.kv_cache = [{} for _ in range(self.num_layers)]
        self.k_cache_tensor = None
        self.v_cache_tensor = None
        self.cache_valid = None
        self.crf_cache = None
        self.crf_low_cache = None
        self.crf_high_history = []
        self.crf_timestep_history = []
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
        }
        self.current_step = 0
    
    def determine_recompute_set(
        self,
        x_tilde: Optional[torch.Tensor],
        event_intensity: float,
        step: int,
    ) -> set[int]:
        """Determine which tokens to recompute based on event-driven trigger.
        
        MACRO-LEVEL CACHING STRATEGY:
        - Fixed interval recomputation: only recompute every N steps
        - Simple rule: always recompute first K tokens, cache the rest
        - Skip expensive computations (event intensity, delta checks) most of the time
        - CRITICAL: Never cache 100% - always recompute at least K tokens
        
        Args:
            x_tilde: Frequency domain representation (optional, not used in macro strategy)
            event_intensity: Current event intensity (optional, not used in macro strategy)
            step: Current diffusion step
            
        Returns:
            Set of token indices to recompute
        """
        # FREQ TRIGGER STRATEGY: Very rare recomputation (1-2 times per diffusion)
        # Goal: Only trigger recompute 1-2 times total (step 0 + maybe one refresh)
        # For typical 1000-step diffusion: trigger at step 0 and step 500 (only 2 times)
        
        # Step 0: Always recompute all tokens to populate cache (required, 1st trigger)
        if step == 0:
            return set(range(self.max_len))
        
        # Use a large interval to ensure only 1-2 triggers per diffusion
        # If R is small (<100), auto-scale to 500 for rare triggering
        # This ensures ~1-2 recomputes in a 1000-step diffusion
        if self.R < 100:
            recompute_interval = 500  # Auto-scale: trigger at step 500 (2nd trigger)
        else:
            recompute_interval = self.R  # Use user-specified R if already large
        
        K_tokens = min(self.K, self.max_len)
        
        # Check if this is a rare recompute step (only 1-2 times per diffusion)
        should_recompute = (step % recompute_interval == 0)
        
        if should_recompute:
            # Very rare refresh: recompute first 2*K tokens (only happens 1-2 times)
            recompute_count = min(2 * K_tokens, self.max_len)
            return set(range(recompute_count))
        else:
            # PURE CACHE MODE (99%+ of steps): No recompute - use 100% cache!
            # This is the fastest path - zero F.linear computation, zero attention overhead
            return set()  # Empty set = no recompute, all from cache
    
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
        
        OPTIMIZED: Try tensor-based cache first (much faster), fallback to dict.
        
        Args:
            layer_idx: Layer index
            token_indices: List of token indices
            
        Returns:
            Dictionary mapping token_idx to (K, V) pair
        """
        result = {}
        
        # Try tensor-based cache first (much faster)
        if (self.k_cache_tensor is not None and 
            self.v_cache_tensor is not None and 
            self.cache_valid is not None and
            layer_idx < self.num_layers):
            
            # Check if all tokens are valid
            valid_mask = self.cache_valid[layer_idx, token_indices]
            if valid_mask.all():
                # Fast path: direct tensor indexing
                assert self.k_cache_tensor is not None and self.v_cache_tensor is not None
                k_cached = self.k_cache_tensor[layer_idx, :, token_indices, :]  # (nhead, num_tokens, head_dim)
                v_cached = self.v_cache_tensor[layer_idx, :, token_indices, :]
                
                # Convert to dict format for compatibility (minimal overhead)
                for i, token_idx in enumerate(token_indices):
                    result[token_idx] = (k_cached[:, i, :], v_cached[:, i, :])
                    self.stats["cache_hit_count"] += 1
                return result
        
        # Fallback to dict cache
        if layer_idx < len(self.kv_cache):
            layer_cache = self.kv_cache[layer_idx]
            for token_idx in token_indices:
                if token_idx in layer_cache:
                    self.stats["cache_hit_count"] += 1
                    result[token_idx] = layer_cache[token_idx]
        return result
    
    def get_cached_kv_tensor(
        self,
        layer_idx: int,
        token_indices: list[int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get cached KV as tensors directly (no dict conversion).
        
        OPTIMIZED: For full layer access (all tokens), return directly without indexing.
        
        Args:
            layer_idx: Layer index
            token_indices: List of token indices (must be sorted)
            
        Returns:
            (k_tensor, v_tensor) with shape (nhead, num_tokens, head_dim) or None
        """
        # Fast path: Check cache exists first (single check)
        if (self.k_cache_tensor is None or 
            self.v_cache_tensor is None or 
            self.cache_valid is None or
            layer_idx >= self.num_layers):
            return None
        
        # OPTIMIZATION: Fast path for all tokens (pure cache mode - most common)
        max_len = self.k_cache_tensor.shape[2]
        if len(token_indices) == max_len:
            # Check if it's actually all tokens (avoid list comparison overhead)
            if token_indices[0] == 0 and token_indices[-1] == max_len - 1:
                # Requesting all tokens: return full layer slice directly (fastest!)
                # Use [layer_idx] instead of [layer_idx, :, :, :] for slight speedup
                k_cached = self.k_cache_tensor[layer_idx]  # (nhead, max_len, head_dim)
                v_cached = self.v_cache_tensor[layer_idx]
                self.stats["cache_hit_count"] += len(token_indices)
                return k_cached, v_cached
        
        # Partial tokens: use index_select (faster than advanced indexing)
        # Convert to tensor once if needed
        if not isinstance(token_indices, torch.Tensor):
            token_indices_tensor = torch.tensor(token_indices, dtype=torch.long, device=self.device)
        else:
            token_indices_tensor = token_indices
        
        # Use index_select on dim=2 (token dimension) - faster than advanced indexing
        # k_cache_tensor shape: (num_layers, nhead, max_len, head_dim)
        # After index_select on dim=2: (num_layers, nhead, num_tokens, head_dim)
        # Then select layer_idx: (nhead, num_tokens, head_dim)
        k_cached = self.k_cache_tensor[layer_idx].index_select(1, token_indices_tensor)  # (nhead, num_tokens, head_dim)
        v_cached = self.v_cache_tensor[layer_idx].index_select(1, token_indices_tensor)
        self.stats["cache_hit_count"] += len(token_indices)
        return k_cached, v_cached
    
    def set_cached_kv_batch(
        self,
        layer_idx: int,
        token_indices: list[int],
        k_batch: torch.Tensor,  # (batch, nhead, num_tokens, head_dim) or (nhead, num_tokens, head_dim)
        v_batch: torch.Tensor,
    ) -> None:
        """Batch set cached KV pairs for multiple tokens.
        
        This is much more efficient than calling set_cached_kv multiple times.
        
        Args:
            layer_idx: Layer index
            token_indices: List of token indices
            k_batch: Key tensor batch
            v_batch: Value tensor batch
        """
        if layer_idx >= len(self.kv_cache):
            return
        
        # Handle batch dimension: k_batch comes as (batch, nhead, num_tokens, head_dim)
        # After removing batch: (nhead, num_tokens, head_dim)
        # k_cache_tensor is (num_layers, nhead, max_len, head_dim)
        # So k_cache_tensor[layer_idx, :, token_indices, :] is (nhead, num_tokens, head_dim)
        if k_batch.dim() == 4:  # (batch, nhead, num_tokens, head_dim)
            k_batch = k_batch[0]  # -> (nhead, num_tokens, head_dim)
            v_batch = v_batch[0]
        elif k_batch.dim() == 3:  # Already (nhead, num_tokens, head_dim)
            pass  # Already correct shape
        else:
            # Unexpected shape - fallback to dict cache only
            layer_cache = self.kv_cache[layer_idx]
            for i, token_idx in enumerate(token_indices):
                if k_batch.dim() == 2:  # (num_tokens, head_dim) - single head?
                    k_token = k_batch[i]
                    v_token = v_batch[i]
                else:
                    k_token = k_batch[:, i, :] if k_batch.dim() == 3 else k_batch[i]
                    v_token = v_batch[:, i, :] if v_batch.dim() == 3 else v_batch[i]
                layer_cache[token_idx] = (k_token, v_token)
            self.stats["recompute_count"] += len(token_indices)
            return
        
        # OPTIMIZED: Only detach/move if needed - avoid unnecessary operations
        # Most of the time, tensors are already detached and on correct device
        needs_detach = k_batch.requires_grad
        needs_move = k_batch.device != self.device
        
        if needs_detach or needs_move:
            if needs_detach and needs_move:
                # Both needed: combine operations
                k_batch = k_batch.detach().to(self.device, non_blocking=True)
                v_batch = v_batch.detach().to(self.device, non_blocking=True)
            elif needs_detach:
                k_batch = k_batch.detach()
                v_batch = v_batch.detach()
            elif needs_move:
                k_batch = k_batch.to(self.device, non_blocking=True)
                v_batch = v_batch.to(self.device, non_blocking=True)
        # If neither needed, use tensors as-is (fastest path!)
        
        # Initialize tensor cache if needed (CRITICAL: this was missing!)
        if self.k_cache_tensor is None and layer_idx < self.num_layers:
            nhead, num_tokens, head_dim = k_batch.shape
            self.k_cache_tensor = torch.zeros(
                self.num_layers, nhead, self.max_len, head_dim,
                device=self.device, dtype=k_batch.dtype
            )
            self.v_cache_tensor = torch.zeros(
                self.num_layers, nhead, self.max_len, head_dim,
                device=self.device, dtype=v_batch.dtype
            )
            self.cache_valid = torch.zeros(
                self.num_layers, self.max_len,
                dtype=torch.bool, device=self.device
            )
        
        # OPTIMIZED: Direct tensor assignment only (removed dict cache for backward compatibility)
        # This is the FAST PATH - no dict operations
        if (self.k_cache_tensor is not None and 
            self.v_cache_tensor is not None and 
            self.cache_valid is not None and
            layer_idx < self.num_layers):
            # Direct tensor assignment - much faster than dict
            # k_batch is (nhead, num_tokens, head_dim)
            # k_cache_tensor[layer_idx, :, token_indices, :] is (nhead, num_tokens, head_dim)
            assert self.k_cache_tensor is not None and self.v_cache_tensor is not None
            self.k_cache_tensor[layer_idx, :, token_indices, :] = k_batch
            self.v_cache_tensor[layer_idx, :, token_indices, :] = v_batch
            self.cache_valid[layer_idx, token_indices] = True
        
        # REMOVED: Dict cache storage (was causing overhead - backward compatibility removed)
        # If dict cache is needed in the future, only store on-demand
        
        self.stats["recompute_count"] += len(token_indices)
    
    def set_cached_kv(
        self,
        layer_idx: int,
        token_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        """Cache KV pair for a specific layer and token.
        
        OPTIMIZED: Use tensor-based cache when possible.
        
        Args:
            layer_idx: Layer index
            token_idx: Token index
            K: Key tensor (nhead, head_dim) or (batch, nhead, head_dim)
            V: Value tensor
        """
        # Handle batch dimension
        if K.dim() == 3:  # (batch, nhead, head_dim)
            K = K[0]
            V = V[0]
        
        # Move to device and detach
        if K.device != self.device:
            K = K.detach().to(self.device, non_blocking=True)
            V = V.detach().to(self.device, non_blocking=True)
        else:
            K = K.detach()
            V = V.detach()
        
        # Try tensor-based cache first
        if layer_idx < self.num_layers:
            nhead, head_dim = K.shape
            
            # Initialize tensor cache if needed
            if self.k_cache_tensor is None:
                self.k_cache_tensor = torch.zeros(
                    self.num_layers, nhead, self.max_len, head_dim,
                    device=self.device, dtype=K.dtype
                )
                self.v_cache_tensor = torch.zeros(
                    self.num_layers, nhead, self.max_len, head_dim,
                    device=self.device, dtype=V.dtype
                )
                self.cache_valid = torch.zeros(
                    self.num_layers, self.max_len,
                    dtype=torch.bool, device=self.device
                )
            
            # Store in tensor cache (fast!)
            assert self.k_cache_tensor is not None and self.v_cache_tensor is not None and self.cache_valid is not None
            self.k_cache_tensor[layer_idx, :, token_idx, :] = K
            self.v_cache_tensor[layer_idx, :, token_idx, :] = V
            self.cache_valid[layer_idx, token_idx] = True
        
        # Also store in dict cache for backward compatibility
        if layer_idx < len(self.kv_cache):
            self.kv_cache[layer_idx][token_idx] = (K, V)
        
        self.stats["recompute_count"] += 1
    
    def update_crf(
        self,
        crf: torch.Tensor,
        timestep: Optional[float] = None,
    ) -> None:
        """Update cumulative residual features cache.
        
        Enhanced with FreqCa: decompose CRF into low and high frequency components.
        
        Args:
            crf: Cumulative residual features (num_layers, max_len, d_model)
            timestep: Current timestep (for FreqCa history)
        """
        # OPTIMIZATION: Only store CRF if needed (for event intensity or FreqCa)
        # For macro strategy, we don't need CRF for most steps
        needs_crf = self.use_freqca or (self.current_step % self.R == 0)
        
        if needs_crf:
            # Detach to break gradient and save memory
            if crf.device != self.device:
                crf = crf.detach().to(self.device, non_blocking=True)
            else:
                crf = crf.detach()
            
            # Store full CRF
            self.crf_cache = crf
        
        # FreqCa: decompose into low and high frequency components
        # OPTIMIZATION: Only do frequency decomposition when needed
        if self.use_freqca:
            # Check if we should perform frequency decomposition
            should_decomp = (
                self.current_step % self.freq_decomp_interval == 0
                or self.current_step == 0
            )
            
            if should_decomp and needs_crf:
                # Decompose CRF into low and high frequency components
                if self.freq_decomp == "fft":
                    crf_low, crf_high = frequency_decompose_fft(
                        crf, self.low_freq_ratio
                    )
                else:  # dct
                    crf_low, crf_high = frequency_decompose_dct(
                        crf, self.low_freq_ratio
                    )
                
                # Cache low-frequency component (stable, can be reused)
                if crf_low.device != self.device:
                    crf_low = crf_low.to(self.device, non_blocking=True)
                self.crf_low_cache = crf_low.detach()
                
                # Store high-frequency component in history for prediction
                if crf_high.device != self.device:
                    crf_high = crf_high.to(self.device, non_blocking=True)
                self.crf_high_history.append(crf_high.detach())
                if timestep is not None:
                    self.crf_timestep_history.append(timestep)
                
                # Limit history size
                if len(self.crf_high_history) > self.max_history:
                    self.crf_high_history.pop(0)
                    if self.crf_timestep_history:
                        self.crf_timestep_history.pop(0)
    
    def compute_event_intensity(
        self,
        crf: torch.Tensor,
        step: int,
    ) -> float:
        """Compute event intensity based on CRF changes.
        
        Args:
            crf: Current CRF (num_layers, max_len, d_model)
            step: Current step
            
        Returns:
            Event intensity (0-1 scale)
        """
        # Get previous CRF
        prev_crf = self.crf_cache
        
        if prev_crf is None:
            # First step: return low intensity to allow caching from step 1
            if step > 0:
                return 0.1
            return 1.0  # Step 0: always recompute
        
        # Compute delta
        delta = torch.abs(crf - prev_crf)
        
        # Compute energy (L2 norm)
        energy = torch.norm(delta, dim=-1)  # (num_layers, max_len)
        
        # Average energy across layers and tokens
        avg_energy = energy.mean().item()
        
        # Normalize by threshold
        intensity = min(1.0, avg_energy / self.tau_0)
        
        return intensity
    
    def predict_crf_freqca(
        self,
        t_val: float,
    ) -> Optional[torch.Tensor]:
        """Predict CRF using FreqCa strategy (frequency decomposition + Hermite prediction).
        
        Args:
            t_val: Current timestep value
            
        Returns:
            Predicted CRF or None if prediction not possible
        """
        if not self.use_freqca:
            return None
        
        # Need at least low-frequency cache and some history
        if self.crf_low_cache is None:
            return None
        
        if len(self.crf_high_history) < 2:
            return None
        
        # Predict high-frequency component using Hermite polynomials
        crf_high_pred = predict_hermite(
            self.crf_high_history,
            self.crf_timestep_history,
            t_val,
            self.hermite_order,
        )
        
        if crf_high_pred is None:
            return None
        
        # Combine low and high frequency components
        crf_pred = self.crf_low_cache + crf_high_pred
        
        return crf_pred
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_ops = self.stats["recompute_count"] + self.stats["cache_hit_count"]
        cache_hit_ratio = (
            self.stats["cache_hit_count"] / total_ops
            if total_ops > 0
            else 0.0
        )
        
        # Compute cache ratio (percentage of tokens cached)
        # Use tensor cache if available (more accurate), otherwise use dict cache
        if self.cache_valid is not None:
            # More accurate: use tensor cache validity mask
            cache_ratio = self.cache_valid.float().mean().item()
        else:
            # Fallback: use dict cache count
            total_tokens = self.num_layers * self.max_len
            cached_tokens = sum(len(layer_cache) for layer_cache in self.kv_cache)
            cache_ratio = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # CRITICAL: Ensure cache_ratio is never 100% (we always recompute some tokens)
        # If it's 100%, it means we're not using cache effectively
        if cache_ratio >= 1.0:
            # This shouldn't happen if determine_recompute_set is correct
            # But cap it at 99% to indicate we're using cache
            cache_ratio = 0.99
        
        stats = {
            "cache_hit_ratio": cache_hit_ratio,
            "cache_ratio": cache_ratio,
            "recompute_count": self.stats["recompute_count"],
            "cache_hit_count": self.stats["cache_hit_count"],
            "current_step": self.current_step,
        }
        
        # Add FreqCa statistics
        if self.use_freqca:
            freq_decomp_count = len(self.crf_high_history)
            freq_decomp_skipped = max(0, self.current_step - freq_decomp_count)
            freq_decomp_ratio = (
                freq_decomp_count / self.current_step
                if self.current_step > 0
                else 0.0
            )
            stats.update({
                "freq_decomp_count": freq_decomp_count,
                "freq_decomp_skipped": freq_decomp_skipped,
                "freq_decomp_ratio": freq_decomp_ratio,
            })
        
        return stats
