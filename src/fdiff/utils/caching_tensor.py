"""Tensor-based E2-CRF caching for efficient acceleration.

This module implements a tensor-based caching mechanism that stores
entire KV tensors instead of individual token KV pairs, significantly
reducing lookup and assembly overhead.
"""

from typing import Optional
import torch


class TensorE2CRFCache:
    """Tensor-based E2-CRF cache for accelerating diffusion model inference.
    
    Key improvements over dict-based cache:
    1. Stores entire KV tensors instead of individual token pairs
    2. Uses tensor slicing instead of dictionary lookups
    3. Batch operations instead of loops
    4. Minimal overhead when cache is not used
    """
    
    def __init__(
        self,
        num_layers: int,
        max_len: int,
        nhead: int,
        head_dim: int,
        device: torch.device,
        K: int = 5,
        R: int = 10,
    ):
        """Initialize tensor-based cache.
        
        Args:
            num_layers: Number of transformer layers
            max_len: Maximum sequence length
            nhead: Number of attention heads
            head_dim: Dimension of each attention head
            device: Device to store cache on
            K: Number of low-frequency tokens to always recompute
            R: Recompute interval (every R steps)
        """
        self.num_layers = num_layers
        self.max_len = max_len
        self.nhead = nhead
        self.head_dim = head_dim
        self.device = device
        self.K = K
        self.R = R
        
        # Tensor-based cache: (num_layers, nhead, max_len, head_dim)
        # None means not cached yet
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        
        # Track which tokens are valid in cache
        self.cache_valid: Optional[torch.Tensor] = None  # (num_layers, max_len) bool
        
        self.current_step = 0
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
        }
    
    def reset(self) -> None:
        """Reset the cache."""
        self.k_cache = None
        self.v_cache = None
        self.cache_valid = None
        self.current_step = 0
        self.stats = {
            "recompute_count": 0,
            "cache_hit_count": 0,
        }
    
    def determine_recompute_set(self, step: int) -> set[int]:
        """Determine which tokens to recompute.
        
        Simple macro strategy:
        - Step 0: recompute all (populate cache)
        - Every R steps: recompute first K tokens
        - Otherwise: recompute first K tokens (rest use cache)
        """
        if step == 0:
            return set(range(self.max_len))
        
        # Always recompute first K tokens
        return set(range(min(self.K, self.max_len)))
    
    def get_cached_kv(
        self,
        layer_idx: int,
        token_indices: list[int],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached KV for tokens (tensor-based, batch operation).
        
        Returns:
            (k_cached, v_cached) or (None, None) if not cached
        """
        if self.k_cache is None or self.cache_valid is None:
            return None, None
        
        # Check if all requested tokens are valid in cache
        if layer_idx >= self.num_layers:
            return None, None
        
        valid_mask = self.cache_valid[layer_idx, token_indices]
        if not valid_mask.all():
            return None, None
        
        # Extract cached KV using tensor slicing (fast!)
        k_cached = self.k_cache[layer_idx, :, token_indices, :]  # (nhead, num_tokens, head_dim)
        v_cached = self.v_cache[layer_idx, :, token_indices, :]
        
        self.stats["cache_hit_count"] += len(token_indices)
        return k_cached, v_cached
    
    def set_cached_kv(
        self,
        layer_idx: int,
        token_indices: list[int],
        k: torch.Tensor,  # (batch, nhead, num_tokens, head_dim) or (nhead, num_tokens, head_dim)
        v: torch.Tensor,
    ) -> None:
        """Cache KV for tokens (tensor-based, batch operation).
        
        Args:
            layer_idx: Layer index
            token_indices: List of token indices to cache
            k: Key tensor
            v: Value tensor
        """
        if layer_idx >= self.num_layers:
            return
        
        # Initialize cache tensors if needed
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                self.num_layers, self.nhead, self.max_len, self.head_dim,
                device=self.device, dtype=k.dtype
            )
            self.v_cache = torch.zeros(
                self.num_layers, self.nhead, self.max_len, self.head_dim,
                device=self.device, dtype=v.dtype
            )
            self.cache_valid = torch.zeros(
                self.num_layers, self.max_len,
                dtype=torch.bool, device=self.device
            )
        
        # Handle batch dimension if present
        if k.dim() == 4:  # (batch, nhead, num_tokens, head_dim)
            k = k[0]  # Take first batch element
            v = v[0]
        
        # Move to device if needed
        if k.device != self.device:
            k = k.to(self.device, non_blocking=True)
            v = v.to(self.device, non_blocking=True)
        
        # Detach to break gradient
        k = k.detach()
        v = v.detach()
        
        # Store in cache using tensor assignment (fast!)
        self.k_cache[layer_idx, :, token_indices, :] = k
        self.v_cache[layer_idx, :, token_indices, :] = v
        self.cache_valid[layer_idx, token_indices] = True
        
        self.stats["recompute_count"] += len(token_indices)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total_ops = self.stats["recompute_count"] + self.stats["cache_hit_count"]
        cache_hit_ratio = (
            self.stats["cache_hit_count"] / total_ops
            if total_ops > 0
            else 0.0
        )
        
        cache_ratio = 0.0
        if self.cache_valid is not None:
            cache_ratio = self.cache_valid.float().mean().item()
        
        return {
            "cache_hit_ratio": cache_hit_ratio,
            "cache_ratio": cache_ratio,
            "recompute_count": self.stats["recompute_count"],
            "cache_hit_count": self.stats["cache_hit_count"],
            "current_step": self.current_step,
        }

