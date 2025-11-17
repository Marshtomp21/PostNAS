import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        layer_idx: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        
        self.key_dim = int(hidden_size * expand_k) // num_heads
        self.value_dim = int(hidden_size * expand_v) // num_heads
        
        self.q_proj = nn.Linear(hidden_size, self.key_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim * num_heads, bias=False)
        
        self.g_proj = nn.Linear(hidden_size, self.value_dim * num_heads, bias=False)
        
        self.o_proj = nn.Linear(self.value_dim * num_heads, hidden_size, bias=False)
        
        self.norm = nn.LayerNorm(self.value_dim * num_heads, eps=1e-5, elementwise_affine=True)
        
        print(f"[GLA] Loaded GLA into Layer {layer_idx}.")
    
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
        
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.key_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.value_dim).transpose(1, 2)
        
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        kv = torch.einsum('bhnd,bhne->bhnde', k, v)
        z = torch.einsum('bhnd,bhnde->bhne', q, kv)
        
        k_sum = k.sum(dim=2, keepdim=True)
        normalizer = torch.einsum('bhnd,bhnd->bhn', q, k_sum).unsqueeze(-1)
        z = z / (normalizer + 1e-6)
        
        z = z.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        z = self.norm(z)
        
        g = F.silu(g)
        z = z * g
        
        output = self.o_proj(z)
        
        return (output, None, None)