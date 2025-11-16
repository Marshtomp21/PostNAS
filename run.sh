#!/bin/bash
# PostNAS 启动脚本
# run.sh

# ====================================================================
# 环境准备
# ====================================================================

# 1. 安装依赖
echo "==========================="
echo "环境准备"
echo "==========================="

# pip install torch transformers datasets accelerate
# pip install flash-linear-attention

# 如果flash-linear-attention安装失败，代码会自动使用fallback实现

# ====================================================================
# 数据准备
# ====================================================================

echo ""
echo "==========================="
echo "数据准备"
echo "==========================="

# 创建数据目录
mkdir -p data/train
mkdir -p data/eval

# 方式1: 使用HuggingFace datasets (推荐)
# 在config.py中设置:
# dataset_name = "allenai/c4"
# streaming = True

# 方式2: 准备本地数据
# 训练数据格式: 每行一个文本样本
# echo "Sample text 1" > data/train/train.txt
# echo "Sample text 2" >> data/train/train.txt

# 评估数据格式(MMLU风格): 每行一个JSON对象
# {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
# echo '{"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}' > data/eval/eval.jsonl

echo "数据准备完成 (请根据实际情况准备数据)"

# ====================================================================
# 单GPU运行 (测试)
# ====================================================================

echo ""
echo "==========================="
echo "单GPU测试运行"
echo "==========================="

# python main.py

# ====================================================================
# 8卡A800运行 (生产)
# ====================================================================

echo ""
echo "==========================="
echo "多卡运行"
echo "==========================="

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29500 \
    main.py

# ====================================================================
# 可选: 使用DeepSpeed加速
# ====================================================================

# 创建DeepSpeed配置文件 ds_config.json:
# {
#   "train_batch_size": 64,
#   "gradient_accumulation_steps": 4,
#   "fp16": {
#     "enabled": false
#   },
#   "bf16": {
#     "enabled": true
#   },
#   "zero_optimization": {
#     "stage": 2,
#     "offload_optimizer": {
#       "device": "cpu"
#     }
#   }
# }

# 使用DeepSpeed启动:
# deepspeed --num_gpus=8 main.py --deepspeed ds_config.json

echo ""
echo "==========================="
echo "运行完成"
echo "==========================="
echo "查看结果:"
echo "  - 训练日志: ./logs/training_log.jsonl"
echo "  - Checkpoints: ./checkpoints/"
echo "  - 搜索结果: ./outputs/beam_search_results.json"
echo "  - 层重要性分析: ./outputs/layer_importance_analysis.json"
echo "==========================="
