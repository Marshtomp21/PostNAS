import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
import os


class BeamSearchPlacement:
    def __init__(
        self,
        super_network,
        eval_dataloader,
        config,
        device: torch.device
    ):
        """
        Args:
            super_network: 训练好的超网络
            eval_dataloader: 评估数据加载器
            config: PostNASConfig配置
            device: 计算设备
        """
        self.super_network = super_network
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.device = device
        self.num_layers = super_network.num_layers
        
        self.super_network.eval()
    
    @torch.no_grad()
    def evaluate_config(
        self,
        layer_config: List[int],
        task: str = "mmlu"
    ) -> float:
        """
        评估特定配置的性能
        
        Args:
            layer_config: 层配置列表,0表示全注意力,1表示线性注意力
            task: 任务类型 ("mmlu", "math", "retrieval")
            
        Returns:
            评估得分(越高越好)
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        layer_choices = [x == 0 for x in layer_config]
        
        for batch_idx, batch in enumerate(tqdm(
            self.eval_dataloader,
            desc=f"Evaluating config",
            leave=False
        )):
            if batch_idx >= self.config.eval_samples // self.config.per_device_eval_batch_size:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            outputs = self.super_network(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_choices=layer_choices,
                return_dict=True
            )
            logits = outputs['logits']
            
            if task == "mmlu":
                labels = batch.get('labels', input_ids[:, 1:])
                labels = labels.to(self.device)
                
                last_token_logits = logits[:, -1, :]
                last_token_labels = labels[:, -1] if labels.dim() > 1 else labels
                
                loss = F.cross_entropy(
                    last_token_logits,
                    last_token_labels,
                    reduction='mean'
                )
                total_loss += loss.item()
                total_samples += input_ids.size(0)
            
            elif task in ["math", "retrieval"]:
                labels = batch.get('labels', input_ids[:, 1:])
                labels = labels.to(self.device)
                
                preds = logits[:, -1, :].argmax(dim=-1)
                last_token_labels = labels[:, -1] if labels.dim() > 1 else labels
                
                correct = (preds == last_token_labels).sum().item()
                total_correct += correct
                total_samples += input_ids.size(0)
        
        if task == "mmlu":
            score = -total_loss / max(total_samples, 1)
        else:
            score = total_correct / max(total_samples, 1)
        
        return score
    
    def search(
        self,
        num_full_attention: int = 2,
        beam_width: int = 3,
        task: str = "mmlu"
    ) -> Tuple[List[int], List[int]]:
        """
        使用Beam Search搜索最优配置
        
        Args:
            num_full_attention: 保留的全注意力层数量
            beam_width: Beam宽度
            task: 搜索任务类型
            
        Returns:
            best_config: 最优配置
            full_attn_layers: 全注意力层的位置列表
        """
        print(f"\n{'='*80}")
        print(f"开始Beam Search")
        print(f"{'='*80}")
        print(f"目标: 保留{num_full_attention}个全注意力层")
        print(f"Beam宽度: {beam_width}")
        print(f"任务: {task}")
        print(f"总层数: {self.num_layers}\n")
        
        beam = [{
            'config': [1] * self.num_layers,
            'score': float('-inf'),
            'history': []
        }]
        
        for step in range(num_full_attention):
            print(f"\n{'='*60}")
            print(f"步骤 {step + 1}/{num_full_attention}")
            print(f"{'='*60}")
            
            candidates = []
            
            for state in beam:
                config = state['config'].copy()
                history = state['history'].copy()
                
                for layer_idx in range(self.num_layers):
                    if config[layer_idx] == 1:
                        new_config = config.copy()
                        new_config[layer_idx] = 0 

                        print(f"评估层 {layer_idx}...", end=' ')
                        score = self.evaluate_config(new_config, task)
                        print(f"得分: {score:.4f}")
                        
                        new_history = history + [layer_idx]
                        candidates.append({
                            'config': new_config,
                            'score': score,
                            'history': new_history,
                            'last_layer': layer_idx
                        })
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = candidates[:beam_width]
            
            print(f"\n当前Beam状态:")
            for i, state in enumerate(beam):
                print(f"  Rank {i+1}: 得分={state['score']:.4f}, "
                      f"层={state['history']}")
        
        best_state = beam[0]
        best_config = best_state['config']
        full_attn_layers = [i for i, x in enumerate(best_config) if x == 0]
        
        print(f"\n{'='*80}")
        print(f"搜索完成!")
        print(f"{'='*80}")
        print(f"最佳得分: {best_state['score']:.4f}")
        print(f"全注意力层位置: {full_attn_layers}")
        print(f"{'='*80}\n")
        
        return best_config, full_attn_layers
    
    def analyze_layer_importance(
        self,
        tasks: List[str] = ["mmlu", "math", "retrieval"]
    ) -> Dict[str, List[float]]:
        """
        分析每一层对不同任务的重要性
        
        Args:
            tasks: 要分析的任务列表
            
        Returns:
            importance_scores: {task: [layer_scores]}
        """
        print(f"\n{'='*80}")
        print(f"分析层重要性")
        print(f"{'='*80}\n")
        
        importance_scores = {}
        
        for task in tasks:
            print(f"\n任务: {task}")
            print(f"{'-'*60}")
            
            scores = []
            
            for layer_idx in range(self.num_layers):
                config = [1] * self.num_layers
                config[layer_idx] = 0
                
                score = self.evaluate_config(config, task)
                scores.append(score)
                
                print(f"层 {layer_idx:2d}: {score:.4f}")
            
            importance_scores[task] = scores
        
        output_file = os.path.join(
            self.config.output_dir,
            "layer_importance_analysis.json"
        )
        with open(output_file, 'w') as f:
            json.dump(importance_scores, f, indent=2)
        
        print(f"\n重要性分析结果已保存至: {output_file}")
        
        return importance_scores
    
    def save_search_results(
        self,
        best_config: List[int],
        full_attn_layers: List[int],
        task: str,
        filename: str = "beam_search_results.json"
    ):
        results = {
            'config': best_config,
            'full_attention_layers': full_attn_layers,
            'task': task,
            'num_layers': self.num_layers,
            'num_full_attention': len(full_attn_layers)
        }
        
        output_path = os.path.join(self.config.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"搜索结果已保存至: {output_path}")


class MultiTaskBeamSearch(BeamSearchPlacement):
    def evaluate_multi_task_config(
        self,
        layer_config: List[int],
        tasks: List[str],
        task_weights: List[float]
    ) -> float:
        """
        评估多任务配置
        
        Args:
            layer_config: 层配置
            tasks: 任务列表
            task_weights: 任务权重
            
        Returns:
            加权得分
        """
        scores = []
        for task in tasks:
            score = self.evaluate_config(layer_config, task)
            scores.append(score)
        
        # 加权平均
        weighted_score = sum(s * w for s, w in zip(scores, task_weights))
        return weighted_score
    
    def search_multi_task(
        self,
        num_full_attention: int = 2,
        beam_width: int = 3,
        tasks: List[str] = ["mmlu", "math", "retrieval"],
        task_weights: List[float] = None
    ) -> Tuple[List[int], List[int]]:
        """
        多任务Beam Search
        
        Args:
            num_full_attention: 全注意力层数
            beam_width: Beam宽度
            tasks: 任务列表
            task_weights: 任务权重
            
        Returns:
            best_config, full_attn_layers
        """
        if task_weights is None:
            task_weights = [1.0 / len(tasks)] * len(tasks)
        
        print(f"\n{'='*80}")
        print(f"多任务Beam Search")
        print(f"{'='*80}")
        print(f"任务: {tasks}")
        print(f"权重: {task_weights}\n")
        
        # 初始化beam
        beam = [{
            'config': [1] * self.num_layers,
            'score': float('-inf'),
            'history': []
        }]
        
        # 迭代搜索
        for step in range(num_full_attention):
            print(f"\n步骤 {step + 1}/{num_full_attention}")
            candidates = []
            
            for state in beam:
                config = state['config'].copy()
                history = state['history'].copy()
                
                for layer_idx in range(self.num_layers):
                    if config[layer_idx] == 1:
                        new_config = config.copy()
                        new_config[layer_idx] = 0
                        
                        # 多任务评估
                        score = self.evaluate_multi_task_config(
                            new_config, tasks, task_weights
                        )
                        
                        new_history = history + [layer_idx]
                        candidates.append({
                            'config': new_config,
                            'score': score,
                            'history': new_history
                        })
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = candidates[:beam_width]
            
            print(f"最佳得分: {beam[0]['score']:.4f}, 层={beam[0]['history']}")
        
        best_config = beam[0]['config']
        full_attn_layers = [i for i, x in enumerate(best_config) if x == 0]
        
        return best_config, full_attn_layers