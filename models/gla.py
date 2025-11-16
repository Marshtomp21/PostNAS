"""
PostNAS - GLA线性注意力实现
models/gla.py

直接使用flash-linear-attention库的GLA实现
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from fla.layers import GatedLinearAttention as FLAGatedLinearAttention


class GatedLinearAttention(nn.Module):
    """
    GLA线性注意力层
    直接使用flash-linear-attention库的优化实现
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_output_gate: bool = True,
        gate_fn: str = "swish",
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        fuse_norm: bool = True,
        layer_idx: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数量
            expand_k: Key维度扩展系数
            expand_v: Value维度扩展系数
            use_short_conv: 是否使用短卷积
            conv_size: 卷积核大小
            use_output_gate: 是否使用输出门控
            gate_fn: 门控激活函数类型
            elementwise_affine: LayerNorm是否使用可学习参数
            norm_eps: LayerNorm的epsilon
            fuse_norm: 是否融合LayerNorm
            layer_idx: 层索引
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        
        # 使用flash-linear-attention的优化实现
        self.gla = FLAGatedLinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            use_output_gate=use_output_gate,
            gate_fn=gate_fn,
            elementwise_affine=elementwise_affine,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            layer_idx=layer_idx,
            **kwargs
        )
        
        print(f"[GLA Layer {layer_idx}] 使用flash-linear-attention库实现")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: KV cache
            output_attentions: 是否输出attention权重
            use_cache: 是否使用cache
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_weights: None (GLA不输出attention权重)
            past_key_value: None或更新的KV cache
        """
        
        # 使用flash-linear-attention实现
        output = self.gla(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        
        # flash-linear-attention可能返回元组或tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
            past_key_value = output[1] if len(output) > 1 else None
            return (hidden_states, None, past_key_value)
        else:
            return (output, None, None)