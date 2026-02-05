#!/usr/bin/env python3
"""Utility functions for model loading and training."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_name: str = "meta-llama/Llama-2-7b-hf", device: str = "cuda"):
    """
    Load model and tokenizer with bf16 precision.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with bf16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable %: {trainable_percent:.2f}%")
    print(f"{'='*60}\n")

    return model, tokenizer


def calculate_token_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate token-level accuracy (ignoring padding tokens with label=-100).

    Args:
        predictions: Predicted token IDs [batch_size, seq_len]
        labels: Ground truth token IDs [batch_size, seq_len]

    Returns:
        Token accuracy as percentage
    """
    # Mask out padding tokens (label=-100)
    mask = labels != -100

    if mask.sum() == 0:
        return 0.0

    # Calculate accuracy only on non-padding tokens
    correct = (predictions == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item() * 100

    return accuracy


def calculate_exact_match(predicted_text: str, ground_truth: str) -> bool:
    """
    Check if predicted text exactly matches ground truth.

    Args:
        predicted_text: Generated text from model
        ground_truth: Ground truth text

    Returns:
        True if exact match, False otherwise
    """
    # Strip whitespace and compare
    return predicted_text.strip() == ground_truth.strip()


def load_checkpoint(checkpoint_path: str, model, optimizer, device: str = "cuda"):
    """
    Load model and optimizer state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        checkpoint dict with epoch, exact_match_count, learned_samples, etc.
    """
    import torch

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    print(f"Previous best exact match: {checkpoint.get('exact_match_count', 0)}")
    if 'learned_samples' in checkpoint:
        print(f"Previously learned samples: {len(checkpoint['learned_samples'])}\n")
    else:
        print()

    return checkpoint
