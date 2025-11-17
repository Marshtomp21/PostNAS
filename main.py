import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
import random
import numpy as np

# 导入自定义模块
from config import PostNASConfig
from models.super_network import OFASuperNetwork
from training.trainer import PostNASTrainer
from training.beam_search import BeamSearchPlacement, MultiTaskBeamSearch
from utils.data_utils import create_dataloaders, prepare_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    else:
        return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def print_config(config: PostNASConfig):
    print("\n" + "="*80)
    print("PostNAS 完整实现 - 配置信息")
    print("="*80)
    print(f"基础模型: {config.base_model_name}")
    print(f"训练步数: {config.max_steps}")
    print(f"学习率: {config.learning_rate}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"梯度累积: {config.gradient_accumulation_steps}")
    print(f"序列长度: {config.max_seq_length}")
    print(f"Beam width: {config.beam_width}")
    print(f"全注意力层数: {config.num_full_attention}")
    print(f"搜索任务: {config.search_task}")
    print(f"混合精度: FP16={config.fp16}, BF16={config.bf16}")
    print("="*80 + "\n")


def step1_train_super_network(
    config: PostNASConfig,
    local_rank: int,
    world_size: int
) -> OFASuperNetwork:
    """
    第一步:训练Once-for-all超网络
    
    Args:
        config: 配置
        local_rank: 本地GPU rank
        world_size: 总GPU数
        
    Returns:
        训练好的超网络
    """
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print("\n" + "="*80)
        print("第一步: 训练Once-for-all超网络")
        print("="*80 + "\n")
    
    tokenizer = prepare_tokenizer(config.base_model_name)
    
    if local_rank == 0:
        print(f"加载基础模型: {config.base_model_name}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map={"": local_rank}
    )
    
    if local_rank == 0:
        print("\n构建超网络...")
    
    super_network = OFASuperNetwork(base_model, freeze_mlp=True)
    
    teacher_model = base_model
    
    if local_rank == 0:
        print("\n加载数据...")
    
    train_dataloader, eval_dataloader = create_dataloaders(
        config,
        tokenizer,
        world_size=world_size,
        rank=local_rank
    )
    
    trainer = PostNASTrainer(
        config=config,
        super_network=super_network,
        teacher_model=teacher_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        local_rank=local_rank
    )
    
    trainer.train()
    
    if local_rank == 0:
        print("\n" + "="*80)
        print("第一步完成: 超网络训练完毕")
        print("="*80 + "\n")
    
    return super_network


def step2_beam_search(
    config: PostNASConfig,
    super_network: OFASuperNetwork,
    local_rank: int
):
    """
    第二步: Beam Search搜索最优全注意力层位置
    
    Args:
        config: 配置
        super_network: 训练好的超网络
        local_rank: 本地GPU rank
    """
    if local_rank != 0:
        return
    
    device = torch.device(f"cuda:{local_rank}")
    
    print("\n" + "="*80)
    print("第二步: Beam Search搜索最优层位置")
    print("="*80 + "\n")
    
    tokenizer = prepare_tokenizer(config.base_model_name)
    
    _, eval_dataloader = create_dataloaders(
        config,
        tokenizer,
        world_size=1,
        rank=0
    )
    
    if eval_dataloader is None:
        print("警告: 没有提供评估数据,跳过Beam Search")
        return
    
    searcher = BeamSearchPlacement(
        super_network=super_network,
        eval_dataloader=eval_dataloader,
        config=config,
        device=device
    )
    
    best_config, full_attn_layers = searcher.search(
        num_full_attention=config.num_full_attention,
        beam_width=config.beam_width,
        task=config.search_task
    )
    
    searcher.save_search_results(
        best_config=best_config,
        full_attn_layers=full_attn_layers,
        task=config.search_task
    )
    
    print("\n执行层重要性分析...")
    importance_scores = searcher.analyze_layer_importance(
        tasks=["mmlu", "math", "retrieval"]
    )
    
    print("\n" + "="*80)
    print("第二步完成: Beam Search搜索完毕")
    print(f"最优全注意力层位置: {full_attn_layers}")
    print("="*80 + "\n")


def main():
    set_seed(42)

    local_rank, world_size = setup_distributed()

    config = PostNASConfig()
    
    if local_rank == 0:
        print_config(config)
    
    try:
        # ====== 第一步: 训练超网络 ======
        super_network = step1_train_super_network(
            config=config,
            local_rank=local_rank,
            world_size=world_size
        )
        
        if dist.is_initialized():
            dist.barrier()
        
        # ====== 第二步: Beam Search ======
        step2_beam_search(
            config=config,
            super_network=super_network.module if isinstance(super_network, DDP) else super_network,
            local_rank=local_rank
        )
        
        if local_rank == 0:
            print("\n" + "="*80)
            print("PostNAS第一步和第二步全部完成!")
            print("="*80)
            print("\n下一步:")
            print("1. 查看输出目录中的搜索结果")
            print("2. 根据最优配置构建最终模型")
            print("="*80 + "\n")
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()