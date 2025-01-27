# Copyright 2023 ByteDance Ltd.
# Licensed under the Apache License, Version 2.0

from accelerate import Accelerator
from dataclasses import dataclass, field, asdict
from datasets import Dataset, DatasetDict
from datetime import timedelta
from functools import partial
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from trl.core import masked_mean, masked_var, masked_whiten, logprobs_from_logits
from typing import List, Dict, Any, Optional
import json
import numpy as np
import os
import random
import torch
import wandb
from peft import LoraConfig

# 全局常量
TIMEOUT = 10
instruction = None
cot_trigger = None
answer_trigger = None

@dataclass
class TrainingConfig:
    """训练配置类"""
    model_name_or_path: str
    tokenizer_name_or_path: str
    model_dir: str
    train_file: str
    test_file: str
    batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    num_epochs: int = 40
    max_input_length: int = 700
    max_gen_length: int = 700
    seed: int = 42
    wandb_log: bool = False
    wandb_project: str = "rl_math"
    wandb_run_name: str = "default_run"
    engine: str = "python"
    vf_coef: float = 1.0
    kl_coef: float = 0.1
    gamma: float = 0.98
    lam: float = 0.95


class CustomDataCollator:
    """自定义数据整理器"""
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 提取基本信息
        ppo_forward_kwargs = {
            'query': [item['prefix_text'] for item in batch],
            'answer_values': [item['answer_value'] for item in batch],
            'item_ids': torch.tensor([int(item['item_id'].split('_')[1]) for item in batch])
        }

        # 处理prefix相关的输入
        prefix_inputs = [{
            'input_ids': item['prefix'],
            'attention_mask': item['prefix_attention_mask']
        } for item in batch]
        
        # 使用DataCollator进行padding
        padded_prefix = self.data_collator(prefix_inputs)

        # 处理labels
        labels = [item['labels'] for item in batch]
        max_label_length = max(len(l) for l in labels)
        padded_labels = torch.full(
            (len(labels), max_label_length),
            -100,
            dtype=torch.long
        )
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = torch.tensor(label)

        generate_prefix_kwargs = {
            'input_ids': padded_prefix['input_ids'],
            'attention_mask': padded_prefix['attention_mask'],
            'labels': padded_labels
        }

        return {
            'ppo_forward_kwargs': ppo_forward_kwargs,
            'generate_prefix_kwargs': generate_prefix_kwargs
        }
        
class CustomPPOTrainer:
    """自定义PPO训练器"""
    def __init__(
        self,
        config: TrainingConfig,
        model: AutoModelForCausalLMWithValueHead,
        ref_model: Optional[AutoModelForCausalLMWithValueHead],
        tokenizer: AutoTokenizer,
        accelerator: Accelerator
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.accelerator = accelerator

    def compute_rewards(
        self,
        model_outputs: torch.Tensor,
        ref_outputs: Optional[torch.Tensor],
        masks: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算奖励和优势值"""
        # 计算KL散度奖励
        kl_rewards = None
        if ref_outputs is not None:
            kl = model_outputs - ref_outputs
            kl_rewards = -kl * masks[:, :-1]
            rewards = rewards + self.config.kl_coef * kl_rewards

        # 计算优势值
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(rewards.size(1))):
            next_values = values[:, t + 1] if t < rewards.size(1) - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * next_values - values[:, t]
            last_gae_lam = delta + self.config.gamma * self.config.lam * last_gae_lam
            advantages[:, t] = last_gae_lam

        # 计算回报值
        returns = advantages + values

        return {
            'advantages': advantages,
            'returns': returns,
            'kl_rewards': kl_rewards
        }

    def train_step(
        self,
        model_inputs: Dict[str, torch.Tensor],
        old_model_outputs: Dict[str, torch.Tensor],
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """执行一步训练"""
        self.model.train()
        
        # 前向传播
        outputs = self.model(**model_inputs)
        logits = outputs.logits
        values = outputs.value
        
        # 计算策略损失
        logprobs = logprobs_from_logits(logits[:, :-1, :], model_inputs['input_ids'][:, 1:])
        ratio = torch.exp(logprobs - old_model_outputs['logprobs'])
        
        # 计算PPO裁剪损失
        pg_losses = -rewards * ratio
        pg_losses2 = -rewards * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        pg_loss = torch.max(pg_losses, pg_losses2).mean()

        # 计算价值损失
        value_loss = 0.5 * ((values - old_model_outputs['returns']) ** 2).mean()

        # 总损失
        loss = pg_loss + self.config.vf_coef * value_loss

        # 反向传播
        self.accelerator.backward(loss)
        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        return {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'value_loss': value_loss.item(),
            'ratio': ratio.mean().item()
        }
        
def prepare_datasets(config: TrainingConfig, tokenizer: AutoTokenizer) -> tuple:
    """准备数据集"""
    # 加载数据
    raw_dataset = DatasetDict({
        'train': Dataset.from_list(json.load(open(config.train_file, 'r'))),
        'test': Dataset.from_list(json.load(open(config.test_file, 'r')))
    })

    # 设置COT提示
    src_name = raw_dataset['train']['item_id'][0].split('_')[0]

    def tokenize_function(examples: Dict[str, List]) -> Dict[str, List]:
        """数据预处理函数"""
        # 构建对话格式的消息列表
        conversations = [
            [
                {"role": "user", "content": f"{question}"},
                {"role": "assistant", "content": f"{answer}"}
            ]
            for question, answer in zip(examples['question'], examples['answer_cot'])
        ]
        
        # 使用chat template处理输入
        formatted_chats = [
            tokenizer.apply_chat_template(
                conversation=conv,
                tokenize=False,
                add_generation_prompt=True
            )
            for conv in conversations
        ]
        
        # tokenize完整对话
        encodings = tokenizer(
            formatted_chats,
            padding=False,
            truncation=True,
            max_length=config.max_input_length,
            return_tensors=None
        )
        
        # 准备labels（将非助手回复的部分mask掉）
        labels = []
        for conv, input_ids in zip(conversations, encodings['input_ids']):
            assistant_reply = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": conv[1]["content"]}],
                tokenize=False,
                add_generation_prompt=False
            )
            assistant_ids = tokenizer.encode(assistant_reply)
            
            label = [-100] * len(input_ids)
            for i in range(len(input_ids) - len(assistant_ids) + 1):
                if input_ids[i:i+len(assistant_ids)] == assistant_ids:
                    label[i:i+len(assistant_ids)] = assistant_ids
                    break
                
            labels.append(label)

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels,
            'query': [conv[0]["content"] for conv in conversations],  # 原始问题文本
            'answer_value': examples['answer_value'],  # ground truth值
            'item_id': examples['item_id']
        }

    # 处理数据集
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
        num_proc=config.num_workers
    )

    return tokenized_dataset, src_name

def main():
    """主函数"""
    # 解析参数
    config = TrainingConfig(
        model_name_or_path="gpt2",  # 替换为实际使用的模型
        tokenizer_name_or_path="gpt2",
        model_dir="./output",
        train_file="path/to/train.json",
        test_file="path/to/test.json"
    )

    # 初始化accelerator
    accelerator = Accelerator()

    # 设置随机种子
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # 初始化wandb
    if accelerator.is_main_process and config.wandb_log:
        wandb.init(project=config.wandb_project, name=config.wandb_run_name)
        wandb.config.update(asdict(config))

    # 初始化LoRA配置
    lora_config = LoraConfig(
        r=16,  # LoRA的秩
        lora_alpha=32,  # LoRA的缩放参数
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 使用PEFT初始化模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name_or_path,
        peft_config=lora_config,
        load_in_8bit=True  # 可选:使用8位精度
    )
    
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name_or_path,
        peft_config=lora_config,
        load_in_8bit=True
    )

    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

    # 准备数据集
    dataset, src_name = prepare_datasets(config, tokenizer)
    
    # 创建数据加载器
    data_collator = CustomDataCollator(tokenizer, config.max_input_length)
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )
    eval_dataloader = DataLoader(
        dataset['test'],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config.num_workers
    )

    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

    # 准备训练
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    if ref_model is not None:
        ref_model = accelerator.prepare(ref_model)

    # 创建训练器
    trainer = CustomPPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator
    )

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # 生成输出
            with torch.no_grad():
                outputs = model(**batch['generate_prefix_kwargs'])
                if ref_model is not None:
                    ref_outputs = ref_model(**batch['generate_prefix_kwargs'])

            # 计算奖励
            rewards_dict = trainer.compute_rewards(
                outputs.logits,
                ref_outputs.logits if ref_model is not None else None,
                batch['generate_prefix_kwargs']['attention_mask'],
                outputs.value,
                torch.ones_like(outputs.value)  # 这里需要根据实际情况计算奖励
            )

            # 训练步骤
            stats = trainer.train_step(
                batch['generate_prefix_kwargs'],
                {
                    'logprobs': logprobs_from_logits(outputs.logits[:, :-1, :], batch['generate_prefix_kwargs']['input_ids'][:, 1:]),
                    'returns': rewards_dict['returns']
                },
                rewards_dict['advantages']
            )

            total_loss += stats['loss']

            # 更新学习率
            scheduler.step()

            # 记录日志
            if accelerator.is_main_process and config.wandb_log:
                wandb.log(stats)

        # 评估
        if epoch % 5 == 0:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    outputs = model(**batch['generate_prefix_kwargs'])
                    eval_loss += outputs.loss.item()

            eval_loss /= len(eval_dataloader)
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch}: train_loss={total_loss/len(train_dataloader):.4f}, eval_loss={eval_loss:.4f}")
                if config.wandb_log:
                    wandb.log({
                        'epoch': epoch,
                        'eval_loss': eval_loss,
                        'train_loss': total_loss/len(train_dataloader)
                    })

        # 保存模型
        if epoch % 10 == 0:
            if accelerator.is_main_process:
                model.save_pretrained(f"{config.model_dir}/checkpoint-{epoch}")
                tokenizer.save_pretrained(f"{config.model_dir}/checkpoint-{epoch}")

if __name__ == "__main__":
    main()