import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import random
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from .gla import GatedLinearAttention


class OFASuperNetwork(nn.Module):
    """
    Once-for-all超网络
    在预训练模型的每个注意力层旁边增加GLA路径
    """
    
    def __init__(self, base_model, freeze_mlp: bool = True):
        """
        Args:
            base_model: 预训练的全注意力模型
            freeze_mlp: 是否冻结MLP权重
        """
        super().__init__()
        
        self.base_model = base_model
        self.config = base_model.config
        self.num_layers = len(base_model.model.layers)
        
        # 为每层添加GLA分支
        self.gla_layers = nn.ModuleList([
            GatedLinearAttention(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                layer_idx=i
            )
            for i in range(self.num_layers)
        ])
        
        # 冻结MLP权重
        if freeze_mlp:
            self._freeze_mlp()
        
        print(f"[SuperNetwork] 初始化完成，共{self.num_layers}层")
        print(f"[SuperNetwork] Hidden size: {self.config.hidden_size}")
        print(f"[SuperNetwork] Num attention heads: {self.config.num_attention_heads}")
    
    def _freeze_mlp(self):
        """冻结所有MLP层的参数"""
        frozen_params = 0
        for layer in self.base_model.model.layers:
            for param in layer.mlp.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"[SuperNetwork] MLP层已冻结，冻结参数: {frozen_params:,}")
    
    def get_trainable_parameters(self):
        """获取可训练参数数量"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[SuperNetwork] 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        return trainable, total
    
    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        use_full_attn: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        单层前向传播
        
        Args:
            layer_idx: 层索引
            hidden_states: 输入hidden states
            use_full_attn: True使用全注意力，False使用GLA
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: KV cache
            output_attentions: 是否输出attention权重
            use_cache: 是否使用cache
            
        Returns:
            hidden_states: 输出hidden states
        """
        layer = self.base_model.model.layers[layer_idx]
        
        # 保存残差
        residual = hidden_states
        
        # Input LayerNorm
        hidden_states = layer.input_layernorm(hidden_states)
        
        if use_full_attn:
            # 使用原始的全注意力
            attn_outputs = layer.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            attn_output = attn_outputs[0]
        else:
            # 使用GLA线性注意力
            attn_outputs = self.gla_layers[layer_idx](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            attn_output = attn_outputs[0]
        
        # 残差连接
        hidden_states = residual + attn_output
        
        # MLP块（冻结的）
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        layer_choices: Optional[List[bool]] = None
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_values: KV cache
            inputs_embeds: 输入embeddings
            labels: 标签（用于计算loss）
            use_cache: 是否使用cache
            output_attentions: 是否输出attention权重
            output_hidden_states: 是否输出所有hidden states
            return_dict: 是否返回字典格式
            layer_choices: 每层的选择，True表示全注意力，False表示GLA
                          如果为None，则随机采样
        
        Returns:
            模型输出
        """
        # 随机采样子网络配置
        if layer_choices is None:
            layer_choices = [random.choice([True, False]) 
                           for _ in range(self.num_layers)]
        
        # 输入embeddings
        if inputs_embeds is None:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
        
        # 准备position_ids
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, inputs_embeds.shape[1],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)
        
        # 准备attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 转换attention_mask为4D格式
        batch_size, seq_length = inputs_embeds.shape[:2]
        attention_mask_4d = self._prepare_attention_mask(
            attention_mask, batch_size, seq_length, inputs_embeds.device
        )
        
        # 逐层前向传播
        hidden_states = inputs_embeds
        for layer_idx in range(self.num_layers):
            hidden_states = self.forward_layer(
                layer_idx,
                hidden_states,
                use_full_attn=layer_choices[layer_idx],
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False
            )
        
        # Final LayerNorm
        hidden_states = self.base_model.model.norm(hidden_states)
        
        # LM Head
        logits = self.base_model.lm_head(hidden_states)
        
        # 计算loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states
            }
        else:
            return (loss, logits, hidden_states) if loss is not None else (logits,)
    
    def _prepare_attention_mask(self, attention_mask, batch_size, seq_length, device):
        """准备4D attention mask"""
        # [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
        attention_mask_4d = attention_mask[:, None, None, :].to(dtype=torch.float32, device=device)
        attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(torch.float32).min
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
            diagonal=1
        )
        causal_mask_4d = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)
        attention_mask_4d = attention_mask_4d.masked_fill(causal_mask_4d, torch.finfo(torch.float32).min)
        
        return attention_mask_4d
