"""Cached Transformer Encoder Layer with KV caching support.

This module implements a transformer encoder layer that supports
selective token computation through KV caching.
"""

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from fdiff.utils.caching import E2CRFCache


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
        
        # If no cache or recomputing all tokens, use standard attention
        if recompute_tokens is None or len(recompute_tokens) == seq_len or self.cache is None:
            # Standard self-attention
            src2 = self.self_attn(src, src, src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Feed-forward
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src
        
        # Performance optimization: if recomputing too many tokens (>80%), 
        # use standard attention (cache overhead not worth it)
        if len(recompute_tokens) > 0.8 * seq_len:
            # Standard self-attention (faster than cache overhead)
            src2 = self.self_attn(src, src, src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Feed-forward
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            
            return src
        
        # Selective computation with KV caching
        # Compute Q for all tokens (needed for attention)
        # Only compute Q, not full QKV to avoid redundant computation
        # Use cached weight slices if available
        if self._q_weight is not None:
            q = F.linear(src, self._q_weight, self._q_bias)
        else:
            q_weight = self.self_attn.in_proj_weight[:self.d_model, :]
            q_bias = self.self_attn.in_proj_bias[:self.d_model] if self.self_attn.in_proj_bias is not None else None
            q = F.linear(src, q_weight, q_bias)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        # Build K and V tensors by selectively computing or using cache
        recompute_list = sorted(recompute_tokens)
        
        # Compute K, V only for tokens that need recomputation
        k_recompute = None
        v_recompute = None
        if len(recompute_list) > 0:
            src_recompute = src[:, recompute_list, :]  # (B, num_recompute, d_model)
            # Only compute KV for recomputed tokens
            # Use cached weight slices if available
            if self._k_weight is not None and self._v_weight is not None:
                assert self._k_weight is not None and self._v_weight is not None  # Type hint for linter
                k_recompute = F.linear(src_recompute, self._k_weight, self._k_bias)
                v_recompute = F.linear(src_recompute, self._v_weight, self._v_bias)
            else:
                k_weight = self.self_attn.in_proj_weight[self.d_model:2*self.d_model, :]
                v_weight = self.self_attn.in_proj_weight[2*self.d_model:, :]
                k_bias = self.self_attn.in_proj_bias[self.d_model:2*self.d_model] if self.self_attn.in_proj_bias is not None else None
                v_bias = self.self_attn.in_proj_bias[2*self.d_model:] if self.self_attn.in_proj_bias is not None else None
                k_recompute = F.linear(src_recompute, k_weight, k_bias)
                v_recompute = F.linear(src_recompute, v_weight, v_bias)
            k_recompute = k_recompute.view(batch_size, len(recompute_list), self.nhead, self.head_dim).transpose(1, 2)  # (B, H, num_recompute, D)
            v_recompute = v_recompute.view(batch_size, len(recompute_list), self.nhead, self.head_dim).transpose(1, 2)
            
            # Cache the recomputed K, V (batch operation)
            if self.cache is not None and self.layer_idx is not None:
                # Batch cache all recomputed tokens at once
                for i, token_idx in enumerate(recompute_list):
                    # Extract single token's KV: (B, H, D)
                    k_token = k_recompute[:, :, i, :]
                    v_token = v_recompute[:, :, i, :]
                    self.cache.set_cached_kv(
                        self.layer_idx,
                        token_idx,
                        k_token,
                        v_token,
                    )
        
        # Build full K and V tensors efficiently
        # Pre-allocate tensors
        k_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
        v_full_tensor = torch.zeros(batch_size, self.nhead, seq_len, self.head_dim, device=src.device, dtype=q.dtype)
        
        # Fill in recomputed tokens (batch operation)
        if len(recompute_list) > 0 and k_recompute is not None and v_recompute is not None:
            k_full_tensor[:, :, recompute_list, :] = k_recompute
            v_full_tensor[:, :, recompute_list, :] = v_recompute
        
        # Fill in cached tokens (batch operation when possible)
        # Optimize: use set difference for faster computation
        if isinstance(recompute_tokens, set):
            cached_tokens = sorted(set(range(seq_len)) - recompute_tokens)
        else:
            recompute_set = set(recompute_tokens)
            cached_tokens = sorted(set(range(seq_len)) - recompute_set)
        
        if len(cached_tokens) > 0 and self.cache is not None and self.layer_idx is not None:
            # Optimize: check if cache device matches src device upfront
            cache_needs_move = self.cache.device != src.device
            
            # Batch retrieve cached values (more efficient)
            cached_kv_dict = self.cache.get_cached_kv_batch(self.layer_idx, cached_tokens)
            
            # Separate cached and missing tokens
            cached_token_indices = []
            missing_indices = []
            cached_k_list = []
            cached_v_list = []
            
            # Collect cached values for batch assignment
            for token_idx in cached_tokens:
                if token_idx in cached_kv_dict:
                    cached_k, cached_v = cached_kv_dict[token_idx]
                    # Only move if device mismatch
                    if cache_needs_move:
                        cached_k = cached_k.to(src.device, non_blocking=True)
                        cached_v = cached_v.to(src.device, non_blocking=True)
                    cached_k_list.append(cached_k)
                    cached_v_list.append(cached_v)
                    cached_token_indices.append(token_idx)
                else:
                    missing_indices.append(token_idx)
            
            # Batch assign cached values using advanced indexing (more efficient)
            if len(cached_k_list) > 0:
                # Stack and assign in one operation
                cached_k_batch = torch.stack(cached_k_list, dim=2)  # (B, H, num_cached, D)
                cached_v_batch = torch.stack(cached_v_list, dim=2)
                # Use advanced indexing for batch assignment
                k_full_tensor[:, :, cached_token_indices, :] = cached_k_batch
                v_full_tensor[:, :, cached_token_indices, :] = cached_v_batch
            
            # Fallback: recompute missing tokens in batch
            if len(missing_indices) > 0:
                src_missing = src[:, missing_indices, :]  # (B, num_missing, d_model)
                # Only compute KV for missing tokens (not Q)
                # Use cached weight slices if available
                if self._k_weight is not None and self._v_weight is not None:
                    assert self._k_weight is not None and self._v_weight is not None  # Type hint for linter
                    k_missing = F.linear(src_missing, self._k_weight, self._k_bias)
                    v_missing = F.linear(src_missing, self._v_weight, self._v_bias)
                else:
                    k_weight = self.self_attn.in_proj_weight[self.d_model:2*self.d_model, :]
                    v_weight = self.self_attn.in_proj_weight[2*self.d_model:, :]
                    k_bias = self.self_attn.in_proj_bias[self.d_model:2*self.d_model] if self.self_attn.in_proj_bias is not None else None
                    v_bias = self.self_attn.in_proj_bias[2*self.d_model:] if self.self_attn.in_proj_bias is not None else None
                    k_missing = F.linear(src_missing, k_weight, k_bias)
                    v_missing = F.linear(src_missing, v_weight, v_bias)
                k_missing = k_missing.view(batch_size, len(missing_indices), self.nhead, self.head_dim).transpose(1, 2)
                v_missing = v_missing.view(batch_size, len(missing_indices), self.nhead, self.head_dim).transpose(1, 2)
                k_full_tensor[:, :, missing_indices, :] = k_missing
                v_full_tensor[:, :, missing_indices, :] = v_missing
        
        # Batch attention computation
        attn_scores = torch.matmul(q, k_full_tensor.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full_tensor)  # (B, H, L, D)
        
        # Reshape and apply output projection
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

