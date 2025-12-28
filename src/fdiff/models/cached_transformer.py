"""Cached Transformer Encoder Layer with KV caching support.

This module implements a transformer encoder layer that supports
selective token computation through KV caching.
"""

from typing import Optional, TYPE_CHECKING
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from fdiff.utils.caching import E2CRFCache

# Global timing tracker for debugging
_timing_stats = {
    "cache_lookup": [],
    "cache_store": [],
    "recompute_linear": [],
    "tensor_assembly": [],
    "attention": [],
    "total_forward": [],
}


class CachedTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with KV caching support for selective token computation.
    
    This layer can skip computation for cached tokens by reusing their KV pairs,
    achieving significant speedup for high-frequency tokens that change slowly.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.head_dim = d_model // nhead
        
        # Self-attention components
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = getattr(F, activation)
        
        # Cache reference (set externally)
        self.cache: Optional["E2CRFCache"] = None
        self.layer_idx: Optional[int] = None
        
        # Cache weight slices to avoid repeated slicing
        self._q_weight: Optional[torch.Tensor] = None
        self._k_weight: Optional[torch.Tensor] = None
        self._v_weight: Optional[torch.Tensor] = None
        self._q_bias: Optional[torch.Tensor] = None
        self._k_bias: Optional[torch.Tensor] = None
        self._v_bias: Optional[torch.Tensor] = None
    
    def set_cache(self, cache: "E2CRFCache", layer_idx: int) -> None:
        """Set cache for this layer.
        
        Args:
            cache: E2CRFCache instance
            layer_idx: Index of this layer
        """
        self.cache = cache
        self.layer_idx = layer_idx
        # Pre-compute weight slices for efficiency
        self._q_weight = self.self_attn.in_proj_weight[:self.d_model, :]
        self._k_weight = self.self_attn.in_proj_weight[self.d_model:2*self.d_model, :]
        self._v_weight = self.self_attn.in_proj_weight[2*self.d_model:, :]
        if self.self_attn.in_proj_bias is not None:
            self._q_bias = self.self_attn.in_proj_bias[:self.d_model]
            self._k_bias = self.self_attn.in_proj_bias[self.d_model:2*self.d_model]
            self._v_bias = self.self_attn.in_proj_bias[2*self.d_model:]
        else:
            self._q_bias = None
            self._k_bias = None
            self._v_bias = None
    
    def forward(
        self,
        src: torch.Tensor,
        recompute_tokens: Optional[set[int]] = None,
    ) -> torch.Tensor:
        """Forward pass with optional KV caching.
        
        Args:
            src: Input tensor (batch_size, seq_len, d_model)
            recompute_tokens: Set of token indices to recompute (None = recompute all)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = src.shape
        
        # If no cache, use standard attention
        if self.cache is None:
            # Standard self-attention
            src2 = self.self_attn(src, src, src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Feed-forward
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src
        
        # If recompute_tokens is None, recompute all tokens
        if recompute_tokens is None:
            recompute_tokens = set(range(seq_len))
        
        # OPTIMIZATION: When recomputing all tokens (e.g., step 0),
        # use standard attention for speed, but still cache the results
        if len(recompute_tokens) == seq_len:
            # Use standard attention (fastest)
            src2 = self.self_attn(src, src, src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Feed-forward
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            # Cache all KV pairs for future use (do this efficiently)
            if self.cache is not None and self.layer_idx is not None:
                # Compute KV for all tokens efficiently
                if self._k_weight is not None and self._v_weight is not None:
                    k_all = F.linear(src, self._k_weight, self._k_bias)
                    v_all = F.linear(src, self._v_weight, self._v_bias)
                else:
                    k_weight = self.self_attn.in_proj_weight[self.d_model:2*self.d_model, :]
                    v_weight = self.self_attn.in_proj_weight[2*self.d_model:, :]
                    k_bias = self.self_attn.in_proj_bias[self.d_model:2*self.d_model] if self.self_attn.in_proj_bias is not None else None
                    v_bias = self.self_attn.in_proj_bias[2*self.d_model:] if self.self_attn.in_proj_bias is not None else None
                    k_all = F.linear(src, k_weight, k_bias)
                    v_all = F.linear(src, v_weight, v_bias)
                
                k_all = k_all.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
                v_all = v_all.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
                
                # Cache all tokens efficiently using batch operation
                if hasattr(self.cache, 'set_cached_kv_batch'):
                    token_indices = list(range(seq_len))
                    self.cache.set_cached_kv_batch(
                        self.layer_idx,
                        token_indices,
                        k_all,  # (batch, nhead, seq_len, head_dim)
                        v_all,
                    )
                else:
                    # Fallback to individual operations
                    for token_idx in range(seq_len):
                        k_token = k_all[:, :, token_idx, :]
                        v_token = v_all[:, :, token_idx, :]
                        self.cache.set_cached_kv(
                            self.layer_idx,
                            token_idx,
                            k_token,
                            v_token,
                        )
            
            return src
        
        # Performance optimization: if recomputing too many tokens (>80%),
        # but NOT all tokens, use standard attention (cache overhead not worth it)
        # However, if recomputing ALL tokens (e.g., step 0), we still need to cache them
        if len(recompute_tokens) > 0.8 * seq_len and len(recompute_tokens) < seq_len:
            # Standard self-attention (faster than cache overhead)
            src2 = self.self_attn(src, src, src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Feed-forward
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src
        
        # OPTIMIZATION: Early check - if recomputing too many tokens, use standard attention
        # This avoids cache overhead when cache wouldn't help
        recompute_ratio = len(recompute_tokens) / seq_len if seq_len > 0 else 0
        if recompute_ratio > 0.8:
            # Recomputing >80% tokens: cache overhead not worth it, use standard attention
            src2 = self.self_attn(src, src, src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src
        
        # OPTIMIZED STRATEGY: Only recompute when recompute_list is non-empty
        # Most steps will have empty recompute_list = 100% cache (fastest path!)
        recompute_list = sorted(recompute_tokens)
        cached_tokens = sorted(set(range(seq_len)) - set(recompute_list))
        
        # SECTION 1: Compute Q (needed for attention) - always computed
        if self._q_weight is not None:
            q = F.linear(src, self._q_weight, self._q_bias)
        else:
            q_weight = self.self_attn.in_proj_weight[:self.d_model, :]
            q_bias = self.self_attn.in_proj_bias[:self.d_model] if self.self_attn.in_proj_bias is not None else None
            q = F.linear(src, q_weight, q_bias)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        # OPTIMIZED PATH: If no recompute needed (most common case), use pure cache
        if len(recompute_list) == 0:
            # PURE CACHE MODE: 100% cache, zero recompute - fastest path!
            # Get all tokens from cache
            all_token_indices = list(range(seq_len))
            if self.cache is not None and self.layer_idx is not None:
                t0 = time.time()
                cached_kv = self.cache.get_cached_kv_tensor(self.layer_idx, all_token_indices)
                if cached_kv is not None:
                    k_cached, v_cached = cached_kv  # (nhead, seq_len, head_dim)
                    # OPTIMIZED: Use expand instead of unsqueeze + view (faster, no copy)
                    # Expand adds batch dimension without copying data
                    k_full_tensor = k_cached.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, nhead, seq_len, head_dim)
                    v_full_tensor = v_cached.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    t1 = time.time()
                    _timing_stats["cache_lookup"].append(t1 - t0)
                else:
                    # Fallback: should not happen if cache is properly initialized
                    k_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
                    v_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
            else:
                k_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
                v_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
        else:
            # MIXED MODE: Some tokens need recompute, some from cache
            # SECTION 2: Pre-allocate K and V tensors
            t0 = time.time()
            k_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
            v_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
            t1 = time.time()
            _timing_stats["tensor_assembly"].append(t1 - t0)
            
            # SECTION 3: CACHE OFFSET - Get cached KV for non-recomputed tokens
            if len(cached_tokens) > 0 and self.cache is not None and self.layer_idx is not None:
                t0 = time.time()
                cached_kv = self.cache.get_cached_kv_tensor(self.layer_idx, cached_tokens)
                if cached_kv is not None:
                    k_cached, v_cached = cached_kv  # (nhead, num_cached, head_dim)
                    cached_indices = torch.tensor(cached_tokens, device=src.device, dtype=torch.long) if not isinstance(cached_tokens, torch.Tensor) else cached_tokens
                    k_full_tensor[:, :, cached_indices, :] = k_cached.unsqueeze(0)
                    v_full_tensor[:, :, cached_indices, :] = v_cached.unsqueeze(0)
                t1 = time.time()
                _timing_stats["cache_lookup"].append(t1 - t0)
            
            # SECTION 4: Recompute KV ONLY for tokens that need recomputation
            t0 = time.time()
            src_recompute = src[:, recompute_list, :]
            if self._k_weight is not None and self._v_weight is not None:
                k_recompute = F.linear(src_recompute, self._k_weight, self._k_bias)
                v_recompute = F.linear(src_recompute, self._v_weight, self._v_bias)
            else:
                k_weight = self.self_attn.in_proj_weight[self.d_model:2*self.d_model, :]
                v_weight = self.self_attn.in_proj_weight[2*self.d_model:, :]
                k_bias = self.self_attn.in_proj_bias[self.d_model:2*self.d_model] if self.self_attn.in_proj_bias is not None else None
                v_bias = self.self_attn.in_proj_bias[2*self.d_model:] if self.self_attn.in_proj_bias is not None else None
                k_recompute = F.linear(src_recompute, k_weight, k_bias)
                v_recompute = F.linear(src_recompute, v_weight, v_bias)
            k_recompute = k_recompute.view(batch_size, len(recompute_list), self.nhead, self.head_dim).transpose(1, 2)
            v_recompute = v_recompute.view(batch_size, len(recompute_list), self.nhead, self.head_dim).transpose(1, 2)
            k_full_tensor[:, :, recompute_list, :] = k_recompute
            v_full_tensor[:, :, recompute_list, :] = v_recompute
            t1 = time.time()
            _timing_stats["recompute_linear"].append(t1 - t0)  # KV recompute (F.linear)
            
            # Cache recomputed KV (simple tensor storage, NO computation)
            if self.cache is not None and self.layer_idx is not None:
                t2 = time.time()
                self.cache.set_cached_kv_batch(self.layer_idx, recompute_list, k_recompute, v_recompute)
                t3 = time.time()
                _timing_stats["cache_store"].append(t3 - t2)
        
        # SECTION 5: Attention computation
        t0 = time.time()
        attn_scores = torch.matmul(q, k_full_tensor.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full_tensor)  # (B, H, L, D)
        t1 = time.time()
        _timing_stats["attention"].append(t1 - t0)
        
        # SECTION 6: Output projection and rest
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = F.linear(attn_output, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)
        
        # Residual connection and normalization
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feed-forward network (apply to all tokens for now)
        # In a full implementation, we could also cache FFN outputs
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


def get_timing_stats() -> dict:
    """Get timing statistics for debugging."""
    stats = {}
    for key, times in _timing_stats.items():
        if times:
            stats[key] = {
                "total": sum(times),
                "mean": sum(times) / len(times),
                "count": len(times),
            }
        else:
            stats[key] = {"total": 0, "mean": 0, "count": 0}
    return stats


def reset_timing_stats():
    """Reset timing statistics."""
    for key in _timing_stats:
        _timing_stats[key] = []

