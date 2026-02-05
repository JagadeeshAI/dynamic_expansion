#!/usr/bin/env python3
"""PyTorch Dataset for sentence triplet training."""

import json
import random
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
import torch


class TripletDataset(Dataset):
    """Dataset for training on sentence triplets."""

    def __init__(self, data_path: str, tokenizer, mode: str = "train", val_size: int = 50, max_samples: int = None):
        """
        Args:
            data_path: Path to JSONL file with triplets
            tokenizer: HuggingFace tokenizer (e.g., Llama tokenizer)
            mode: "train" or "val"
            val_size: Number of samples to use for validation
            max_samples: For curriculum learning - use only first N samples (None = use all)
        """
        self.tokenizer = tokenizer
        self.mode = mode

        # Load all triplets
        with open(data_path, 'r') as f:
            all_data = [json.loads(line) for line in f]

        # Split train/val
        # Validation is a SUBSET of training (to test memorization)
        random.seed(42)
        val_indices = set(random.sample(range(len(all_data)), min(val_size, len(all_data))))

        if mode == "train":
            # Train on ALL data (or first max_samples for curriculum learning)
            if max_samples is not None:
                self.data = all_data[:max_samples]
            else:
                self.data = all_data
        else:  # val
            # Validation is a subset of training data
            self.data = [d for i, d in enumerate(all_data) if i in val_indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        s1, s2, s3 = item['s1'], item['s2'], item['s3']

        if self.mode == "train":
            # Training: "given s1: {s1} and s2: {s2} predict s3: {s3}"
            prompt = f"given s1: {s1} and s2: {s2} predict s3: {s3}"
            # Tokenize full sequence
            encoding = self.tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
                padding=False,
                truncation=False
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            # Labels are same as input_ids for causal LM
            labels = input_ids.clone()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'length': len(input_ids)
            }

        else:  # validation
            # Validation: "given s1: {s1} and s2: {s2} predict s3: "
            prompt = f"given s1: {s1} and s2: {s2} predict s3: "

            # Tokenize prompt (without s3)
            prompt_encoding = self.tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
                padding=False,
                truncation=False
            )

            # Tokenize ground truth s3 for reference
            s3_encoding = self.tokenizer(
                s3,
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
                truncation=False
            )

            return {
                'input_ids': prompt_encoding['input_ids'].squeeze(0),
                'attention_mask': prompt_encoding['attention_mask'].squeeze(0),
                'ground_truth': s3,
                'ground_truth_ids': s3_encoding['input_ids'].squeeze(0),
                'length': len(prompt_encoding['input_ids'].squeeze(0))
            }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function with dynamic padding based on max length in batch.
    Batch size should be 1, but this handles dynamic length properly.
    """
    # Find max length in this batch
    max_len = max(item['length'] for item in batch)

    # Prepare batch tensors
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        # Pad to max_len
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), 0, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

        # Handle labels for training
        if 'labels' in item:
            labels = item['labels']
            if pad_len > 0:
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
            labels_list.append(labels)

    result = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
    }

    if labels_list:
        result['labels'] = torch.stack(labels_list)

    # Pass through ground truth for validation
    if 'ground_truth' in batch[0]:
        result['ground_truth'] = [item['ground_truth'] for item in batch]
        result['ground_truth_ids'] = [item['ground_truth_ids'] for item in batch]

    return result


def get_dataloaders(data_path: str, tokenizer, batch_size: int = 1, val_size: int = 50, max_samples: int = None):
    """
    Create train and validation dataloaders.

    Args:
        data_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size (default=1 as requested)
        val_size: Number of validation samples
        max_samples: For curriculum learning - limit training to first N samples

    Returns:
        train_loader, val_loader
    """
    train_dataset = TripletDataset(data_path, tokenizer, mode="train", val_size=val_size, max_samples=max_samples)
    val_dataset = TripletDataset(data_path, tokenizer, mode="val", val_size=val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, val_loader

