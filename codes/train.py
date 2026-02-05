#!/usr/bin/env python3
"""Training script for sentence triplet prediction."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import argparse

from data import get_dataloaders
from utils import get_model, calculate_token_accuracy, calculate_exact_match, load_checkpoint


def train_epoch(model, train_loader, optimizer, device, epoch, tokenizer):
    """Train for one epoch and track which samples are learned."""
    model.train()
    total_loss = 0
    total_token_acc = 0
    exact_match_count = 0
    num_batches = 0
    learned_sample_ids = set()  # Track which samples got exact match

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        # Shift logits and labels for next-token prediction
        # logits[:, :-1, :] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        predictions = torch.argmax(shift_logits, dim=-1)
        token_acc = calculate_token_accuracy(predictions, shift_labels)

        # Check if this sample achieved exact match during training
        with torch.no_grad():
            pred_tokens = predictions[0]
            label_tokens = shift_labels[0]
            mask = label_tokens != -100
            if mask.sum() > 0:
                matches = (pred_tokens[mask] == label_tokens[mask]).all()
                if matches:
                    exact_match_count += 1
                    learned_sample_ids.add(batch_idx)

        # Update running stats
        total_loss += loss.item()
        total_token_acc += token_acc
        num_batches += 1


    avg_loss = total_loss / num_batches
    avg_token_acc = total_token_acc / num_batches

    return avg_loss, avg_token_acc, learned_sample_ids


def evaluate(model, val_loader, tokenizer, device, epoch, learned_sample_ids=None):
    """Evaluate on validation set, optionally only on learned samples."""

    # Skip evaluation if no samples have been learned yet
    if learned_sample_ids is not None and len(learned_sample_ids) == 0:
        print(f"No samples learned yet - skipping evaluation\n")
        return 0.0, 0

    model.eval()
    total_loss = 0
    total_token_acc = 0
    exact_match_count = 0
    num_batches = 0
    tested_samples = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [EVAL]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # If we're only evaluating learned samples, skip others
            if learned_sample_ids is not None and batch_idx not in learned_sample_ids:
                continue

            tested_samples += 1

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ground_truth = batch['ground_truth'][0]  # batch_size=1


            # Generate prediction
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode prediction (remove prompt)
            prompt_length = input_ids.shape[1]
            generated_text = tokenizer.decode(
                generated_ids[0][prompt_length:],
                skip_special_tokens=True
            ).strip()

            # Check exact match
            is_exact_match = calculate_exact_match(generated_text, ground_truth)
            if is_exact_match:
                exact_match_count += 1

            # Calculate token accuracy on ground truth
            ground_truth_ids = batch['ground_truth_ids'][0].to(device)

            # Build full sequence: prompt + ground_truth for teacher forcing
            full_input_ids = torch.cat([input_ids[0], ground_truth_ids], dim=0).unsqueeze(0)
            full_attention_mask = torch.ones_like(full_input_ids)

            # Get logits for full sequence
            outputs = model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask
            )

            # Extract logits that predict ground_truth tokens
            # logits at position i predict token at position i+1
            # So logits[prompt_len-1:prompt_len+gt_len-1] predict ground_truth_ids
            prompt_len = input_ids.shape[1]
            gt_len = len(ground_truth_ids)

            # Get the logits that predict the ground truth tokens
            pred_logits = outputs.logits[0, prompt_len-1:prompt_len+gt_len-1, :]
            predictions = torch.argmax(pred_logits, dim=-1)

            # Ensure same length
            min_len = min(len(predictions), len(ground_truth_ids))
            predictions = predictions[:min_len]
            ground_truth_ids_trimmed = ground_truth_ids[:min_len]

            token_acc = calculate_token_accuracy(predictions.unsqueeze(0), ground_truth_ids_trimmed.unsqueeze(0))

            total_token_acc += token_acc
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'token_acc': f'{token_acc:.2f}%',
                'exact_match': f'{exact_match_count}/{tested_samples}',
                'avg_token_acc': f'{total_token_acc/num_batches:.2f}%'
            })

    avg_token_acc = total_token_acc / num_batches if num_batches > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Evaluation Results:")
    print(f"Tested Samples: {tested_samples} (only learned samples)")
    print(f"Token Accuracy: {avg_token_acc:.2f}%")
    print(f"Exact Match Count: {exact_match_count}/{tested_samples}")
    print(f"{'='*60}\n")

    return avg_token_acc, exact_match_count


def main(args):
    """Main training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer = get_model(args.model_name, device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1

    # Get total number of samples
    import json
    with open(args.data_path, 'r') as f:
        total_samples = sum(1 for _ in f)

    print(f"Total samples in dataset: {total_samples}")
    print(f"One epoch = training on [0], [0,1], [0,1,2], ..., [0,1,...,{total_samples-1}]\n")

    # Track learned samples across all epochs
    all_learned_samples = set()
    best_exact_match = 0

    # Training loop - each epoch goes through full curriculum
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}\n")

        # Reset learned samples for this epoch
        epoch_learned_samples = set()

        # Bidirectional curriculum learning:
        # Odd epochs (1,3,5,...): forward 1→N
        # Even epochs (2,4,6,...): backward N→1
        if epoch % 2 == 1:
            # Odd epoch: forward
            curriculum_range = range(1, total_samples + 1)
            direction = "forward (1→N)"
        else:
            # Even epoch: backward
            curriculum_range = range(total_samples, 0, -1)
            direction = "backward (N→1)"

        print(f"Curriculum direction: {direction}\n")

        # One epoch = curriculum from 1 sample to N samples (or reverse)
        curriculum_pbar = tqdm(curriculum_range, desc=f"Epoch {epoch}/{args.epochs}")
        for curriculum_step in curriculum_pbar:

            # Get dataloaders for this curriculum step
            train_loader, val_loader = get_dataloaders(
                data_path=args.data_path,
                tokenizer=tokenizer,
                batch_size=1,
                val_size=args.val_size,
                max_samples=curriculum_step
            )
            # Train on this curriculum step
            train_loss, train_token_acc, learned_sample_ids = train_epoch(
                model, train_loader, optimizer, device,
                f"{epoch}.{curriculum_step}", tokenizer
            )

            # Accumulate learned samples for this epoch
            epoch_learned_samples.update(learned_sample_ids)
            all_learned_samples.update(learned_sample_ids)

            # Update curriculum progress bar
            curriculum_pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'token_acc': f'{train_token_acc:.2f}%',
                'learned': len(epoch_learned_samples)
            })


        # After completing full curriculum (one epoch), evaluate
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} COMPLETED - Evaluating on learned samples")
        print(f"{'='*60}\n")

        if len(epoch_learned_samples) > 0:
            # Get validation loader with all samples for evaluation
            _, val_loader = get_dataloaders(
                data_path=args.data_path,
                tokenizer=tokenizer,
                batch_size=1,
                val_size=args.val_size,
                max_samples=None  # All samples
            )

            print(f"Evaluating on {len(epoch_learned_samples)} learned samples from this epoch...")
            val_token_acc, exact_match_count = evaluate(
                model, val_loader, tokenizer, device, epoch,
                learned_sample_ids=epoch_learned_samples
            )
        else:
            print(f"No samples learned yet - skipping evaluation\n")
            val_token_acc = 0.0
            exact_match_count = 0

        # Save checkpoint after each epoch
        checkpoint_path = f"{args.output_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_token_acc': val_token_acc,
            'exact_match_count': exact_match_count,
            'learned_samples': list(all_learned_samples)
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model if exact match improved
        if exact_match_count > best_exact_match:
            best_exact_match = exact_match_count
            best_model_path = f"{args.output_dir}/best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_token_acc': val_token_acc,
                'exact_match_count': exact_match_count,
                'learned_samples': list(all_learned_samples)
            }, best_model_path)
            print(f"✓ New best model saved! Exact match: {exact_match_count} (improved from {best_exact_match - exact_match_count})")
        print()

    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentence triplet model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name")
    parser.add_argument("--data_path", type=str, default="data/data.jsonl", help="Path to data JSONL")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--val_size", type=int, default=50, help="Validation set size")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints/checkpoint_epoch_1.pt)")

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
