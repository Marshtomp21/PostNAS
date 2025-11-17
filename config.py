import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PostNASConfig:
    
    # ========== 模型配置 ==========
    base_model_name: str = "/public/liguoqi/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    teacher_model_name: Optional[str] = None  # 如果为None,使用base_model作为教师
    
    # ========== 训练配置 ==========
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Batch配置
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    
    # 序列长度
    max_seq_length: int = 2048
    
    # ========== 训练数据配置 ==========
    train_data_path: str = "./data/train"
    num_workers: int = 4
    
    # 训练数据集 - 通用预训练数据
    dataset_name: Optional[str] = "allenai/c4"
    dataset_config: Optional[str] = "en"
    streaming: bool = True
    
    # ========== 评估数据配置 ==========
    eval_data_path: str = "./data/eval"
    
    # 评估数据集 - 使用MMLU
    eval_dataset_name: Optional[str] = "cais/mmlu"
    eval_dataset_config: Optional[str] = "all"
    eval_dataset_split: str = "test"  # MMLU使用test或validation划分
    
    # ========== 分布式配置 ==========
    world_size: int = 8  # 8卡A800
    
    # ========== Beam Search配置 ==========
    beam_width: int = 3
    num_full_attention: int = 2
    eval_samples: int = 1000  # 用于beam search的评估样本数
    
    # 搜索目标任务
    search_task: str = "mmlu"  # "mmlu", "math", "retrieval"
    
    # ========== 线性注意力块选择配置 ==========
    linear_attention_candidates: List[str] = field(default_factory=lambda: [
        "retnet", "mamba2", "gla", "deltanet", "gated_deltanet"
    ])
    
    # ========== 路径配置 ==========
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # 保存配置
    save_steps: int = 5000
    save_total_limit: int = 3
    
    # 日志配置
    logging_steps: int = 100
    eval_steps: int = 1000
    
    # ========== 蒸馏配置 ==========
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    
    # ========== 混合精度配置 ==========
    fp16: bool = False
    bf16: bool = True
    
    # ========== 其他配置 ==========
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 如果teacher_model_name为None,使用base_model
        if self.teacher_model_name is None:
            self.teacher_model_name = self.base_model_name