import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional, Dict, List
import os


class TextDataset(Dataset):
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
        
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
        else:
            dataset = load_dataset(data_path, split=split, streaming=True)
            self.texts = []
            for i, example in enumerate(dataset):
                if i >= 10000:
                    break
                text = example.get('text', '')
                if text:
                    self.texts.append(text)
        
        print(f"[Dataset] 加载{len(self.texts)}个样本")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
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

class StreamingDatasetWrapper(IterableDataset):
    
    def __init__(self, streaming_dataset):
        super().__init__()
        self.streaming_dataset = streaming_dataset
    
    def __iter__(self):
        return iter(self.streaming_dataset)

class MMEvalDataset(Dataset): 
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
        
        # {"question": "...", "choices": ["A", "B", "C", "D"], "answer": 0}
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
        
        question = example['question']
        choices = example['choices']
        answer = example['answer']
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
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
    if config.streaming and config.dataset_name:
        streaming_dataset = StreamingTextDataset(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            split="train"
        )
        
        train_dataset = StreamingDatasetWrapper(streaming_dataset)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_dataset = TextDataset(
            data_path=config.train_data_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            split="train"
        )
        
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
    
    eval_dataloader = None
    if config.eval_data_path and os.path.exists(config.eval_data_path):
        eval_dataset = MMEvalDataset(
            data_path=config.eval_data_path,
            tokenizer=tokenizer,
            max_length=config.max_seq_length
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer