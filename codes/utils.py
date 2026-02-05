#!/usr/bin/env python3
"""Utility functions for model loading and training."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_name: str = "meta-llama/Llama-2-7b-hf", device: str = "cuda"):
    """
    Load model and tokenizer with bf16 precision.
    Freeze all layers except FFNs.

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

    # Freeze input embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False
        print("✓ Froze input embedding layer (model.model.embed_tokens)")

    # Freeze LM head (output projection)
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = False
        print("✓ Froze output LM head (model.lm_head)")

    # Freeze all layers except FFNs
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            # Freeze attention layers
            if hasattr(layer, 'self_attn'):
                for param in layer.self_attn.parameters():
                    param.requires_grad = False

            # Freeze layer norms
            if hasattr(layer, 'input_layernorm'):
                for param in layer.input_layernorm.parameters():
                    param.requires_grad = False
            if hasattr(layer, 'post_attention_layernorm'):
                for param in layer.post_attention_layernorm.parameters():
                    param.requires_grad = False

            # Keep FFN trainable (mlp)
            if hasattr(layer, 'mlp'):
                for param in layer.mlp.parameters():
                    param.requires_grad = True

        print("✓ Froze all attention and norm layers")
        print("✓ Kept FFN (mlp) layers trainable")

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable %: {trainable_percent:.2f}%")
    print(f"Reduction: {100 - trainable_percent:.2f}% of parameters frozen")
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
    Check if predicted text matches ground truth (relaxed: ignores punctuation and extra whitespace).

    Args:
        predicted_text: Generated text from model
        ground_truth: Ground truth text

    Returns:
        True if match (ignoring punctuation/whitespace), False otherwise
    """
    import re
    
    def normalize_text(text: str) -> str:
        """Normalize text by removing punctuation and extra whitespace, keeping only letters and numbers."""
        # Remove punctuation and keep only alphanumeric characters and spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace (multiple spaces/tabs/newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        # Strip and convert to lowercase for comparison
        return text.strip().lower()
    
    normalized_pred = normalize_text(predicted_text)
    normalized_truth = normalize_text(ground_truth)
    
    return normalized_pred == normalized_truth


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
