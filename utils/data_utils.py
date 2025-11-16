import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional, Dict, List
import os


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        split: str = "train"
    ):
        """
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大序列长度
            split: 数据集划分
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        if os.path.isfile(data_path):
            # 从本地文件加载
            with open(data_path, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
        else:
            # 从HuggingFace加载
            dataset = load_dataset(data_path, split=split, streaming=True)
            self.texts = []
            for i, example in enumerate(dataset):
                if i >= 10000:  # 限制大小用于测试
                    break
                text = example.get('text', '')
                if text:
                    self.texts.append(text)
        
        print(f"[Dataset] 加载{len(self.texts)}个样本")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 分词
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


class StreamingTextDataset:
    """流式文本数据集(用于大规模数据)"""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        tokenizer,
        max_length: int = 2048,
        split: str = "train"
    ):
        """
        Args:
            dataset_name: HuggingFace数据集名称
            dataset_config: 数据集配置
            tokenizer: 分词器
            max_length: 最大序列长度
            split: 数据集划分
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载流式数据集
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True
        )
        
        print(f"[StreamingDataset] 加载流式数据集: {dataset_name}")
    
    def __iter__(self):
        for example in self.dataset:
            text = example.get('text', '')
            if not text:
                continue
            
            # 分词
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            yield {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }


class MMEvalDataset(Dataset):
    """多选题评估数据集(用于MMLU等)"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048
    ):
        """
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载MMLU风格的数据
        # 格式: {"question": "...", "choices": ["A", "B", "C", "D"], "answer": 0}
        import json
        self.examples = []
        
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        
        print(f"[MMEvalDataset] 加载{len(self.examples)}个样本")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 构造prompt
        question = example['question']
        choices = example['choices']
        answer = example['answer']
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # 分词
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(answer, dtype=torch.long)
        }


def create_dataloaders(
    config,
    tokenizer,
    world_size: int = 1,
    rank: int = 0
) -> tuple:
    """
    创建训练和评估数据加载器
    
    Args:
        config: PostNASConfig配置
        tokenizer: 分词器
        world_size: 分布式训练world size
        rank: 当前进程rank
        
    Returns:
        train_dataloader, eval_dataloader
    """
    # 训练数据集
    if config.streaming and config.dataset_name:
        # 流式数据集
        train_dataset = StreamingTextDataset(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            split="train"
        )
        
        # 流式数据集不需要sampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            num_workers=config.num_workers
        )
    else:
        # 常规数据集
        train_dataset = TextDataset(
            data_path=config.train_data_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            split="train"
        )
        
        # 分布式sampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if world_size > 1 else None
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # 评估数据集
    eval_dataloader = None
    if config.eval_data_path and os.path.exists(config.eval_data_path):
        # 检查是否是多选题数据集
        is_mc = config.search_task in ["mmlu", "math", "retrieval"]
        
        if is_mc:
            eval_dataset = MMEvalDataset(
                data_path=config.eval_data_path,
                tokenizer=tokenizer,
                max_length=config.max_seq_length
            )
        else:
            eval_dataset = TextDataset(
                data_path=config.eval_data_path,
                tokenizer=tokenizer,
                max_length=config.max_seq_length,
                split="validation"
            )
        
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        ) if world_size > 1 else None
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.per_device_eval_batch_size,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    return train_dataloader, eval_dataloader


def prepare_tokenizer(model_name: str):
    """
    准备tokenizer
    
    Args:
        model_name: 模型名称
        
    Returns:
        tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer
