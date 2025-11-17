import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
from typing import Dict, Optional, List
import random
import numpy as np


class PostNASTrainer:
    def __init__(
        self,
        config,
        super_network,
        teacher_model,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        local_rank: int = 0
    ):
        """
        Args:
            config: PostNASConfig配置对象
            super_network: OFASuperNetwork超网络
            teacher_model: 教师模型
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            local_rank: 本地GPU rank
        """
        self.config = config
        self.super_network = super_network
        self.teacher_model = teacher_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.local_rank = local_rank
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.device = torch.device(f"cuda:{local_rank}")
        
        self.super_network = self.super_network.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        self.optimizer_params = []
        for gla in self.super_network.gla_layers:
            self.optimizer_params.extend(list(gla.parameters()))
        
        self.optimizer = torch.optim.AdamW(
            self.optimizer_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = config.max_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.scaler = GradScaler() if config.fp16 else None
        self.use_amp = config.fp16 or config.bf16
        self.amp_dtype = torch.float16 if config.fp16 else torch.bfloat16
        
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        self.log_history = []
        
        if self.is_main_process():
            trainable, total = self.super_network.get_trainable_parameters()
            print(f"[Trainer] 可训练参数: {trainable:,} / {total:,}")
            print(f"[Trainer] 优化器参数组: {len(self.optimizer.param_groups)}")
            print(f"[Trainer] 总训练步数: {total_steps}")
            print(f"[Trainer] Warmup步数: {config.warmup_steps}")
    
    def is_main_process(self) -> bool:
        return self.global_rank == 0
    
    def train(self):
        self.super_network.train()
        
        if self.is_main_process():
            print(f"\n{'='*80}")
            print(f"开始训练")
            print(f"{'='*80}\n")
        
        progress_bar = tqdm(
            total=self.config.max_steps,
            desc="Training",
            disable=not self.is_main_process()
        )
        
        accumulated_loss = 0.0
        self.optimizer.zero_grad()
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.optimizer_params,
                            self.config.max_grad_norm
                        )
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    accumulated_loss = 0.0
                    
                    if (self.global_step + 1) % self.config.logging_steps == 0:
                        self.log_metrics({
                            'train/loss': avg_loss,
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/epoch': self.epoch,
                            'train/global_step': self.global_step + 1
                        })
                    
                    if self.eval_dataloader is not None and \
                       (self.global_step + 1) % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self.log_metrics(eval_metrics)
                        self.super_network.train()
                    
                    if (self.global_step + 1) % self.config.save_steps == 0:
                        self.save_checkpoint()
                
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                if self.global_step >= self.config.max_steps:
                    break
            
            self.epoch += 1
        
        progress_bar.close()
        
        # 最终保存
        if self.is_main_process():
            self.save_checkpoint(is_final=True)
            print(f"\n{'='*80}")
            print(f"训练完成！")
            print(f"总步数: {self.global_step}")
            print(f"{'='*80}\n")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单步训练
        
        Args:
            batch: 批次数据
            
        Returns:
            loss值
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        layer_choices = [random.choice([True, False]) 
                        for _ in range(self.super_network.num_layers)]
        
        with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            student_outputs = self.super_network(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_choices=layer_choices,
                return_dict=True
            )
            student_logits = student_outputs['logits']
            
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            loss = self.compute_distillation_loss(
                student_logits,
                teacher_logits,
                labels=input_ids
            )
            
            loss = loss / self.config.gradient_accumulation_steps
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            labels: 真实标签
            
        Returns:
            蒸馏损失
        """
        temperature = self.config.distillation_temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        if labels is not None and self.config.distillation_alpha < 1.0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            loss = self.config.distillation_alpha * kl_loss + \
                   (1 - self.config.distillation_alpha) * ce_loss
        else:
            loss = kl_loss
        
        return loss
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.super_network.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", 
                         disable=not self.is_main_process()):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            layer_choices = [False] * self.super_network.num_layers
            
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                student_outputs = self.super_network(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer_choices=layer_choices,
                    return_dict=True
                )
                student_logits = student_outputs['logits']
                
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
                
                loss = self.compute_distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels=input_ids
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        metrics = {
            'eval/loss': avg_loss,
            'eval/perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float]):
        if not self.is_main_process():
            return
        
        self.log_history.append(metrics)
        
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Step {self.global_step}] {log_str}")
        
        log_file = os.path.join(self.config.log_dir, "training_log.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def save_checkpoint(self, is_final: bool = False):
        if not self.is_main_process():
            return
        
        checkpoint_name = f"checkpoint-{self.global_step}" if not is_final else "final"
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if isinstance(self.super_network, DDP):
            model_state = self.super_network.module.state_dict()
        else:
            model_state = self.super_network.state_dict()
        
        torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))
        
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss
        }
        if self.scaler is not None:
            training_state['scaler'] = self.scaler.state_dict()
        
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        import json
        from dataclasses import asdict
        with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"[Trainer] Checkpoint保存至: {checkpoint_dir}")
        
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        if self.config.save_total_limit <= 0:
            return
        
        checkpoints = []
        for name in os.listdir(self.config.checkpoint_dir):
            if name.startswith("checkpoint-"):
                step = int(name.split("-")[1])
                checkpoints.append((step, name))
        
        checkpoints.sort(reverse=True)
        
        for _, name in checkpoints[self.config.save_total_limit:]:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
            import shutil
            shutil.rmtree(checkpoint_path)
            print(f"[Trainer] 删除旧checkpoint: {checkpoint_path}")