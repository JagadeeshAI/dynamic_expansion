"""Train MMLU model with validation subset evaluation."""

import os
import json
import torch
import random
from tqdm import tqdm
from torch.utils.data import Subset
from codes.data import get_dataloader, MMLUDataset
from codes.utils import get_model, calculate_exact_match


def train_epoch(model, tokenizer, loader, optimizer, device, eos_weight=3.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    exact_matches = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        question = batch['question'][0]
        answer = batch['answer'][0]

        # Format prompt - add EOS token to answer for proper stopping
        prompt = f"Question: {question}\nAnswer:"
        full_text = f"{prompt} {answer}{tokenizer.eos_token}"

        # Calculate max_length dynamically based on this sample
        max_length = len(tokenizer(full_text, return_tensors="pt")['input_ids'][0]) + 10  # +10 buffer

        # Tokenize with dynamic max_length
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
        prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        prompt_len = len(prompt_inputs['input_ids'][0])

        # Calculate max_new_tokens for generation
        max_new_tokens = max_length - prompt_len

        # Create labels (only compute loss on answer tokens including EOS)
        labels = inputs['input_ids'].clone()
        labels[:, :prompt_len] = -100

        # Forward pass with weighted loss
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

        # Calculate weighted cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Create loss weights with EOS weight
        loss_weights = torch.ones_like(shift_labels, dtype=torch.float)
        eos_positions = (shift_labels == tokenizer.eos_token_id)
        loss_weights[eos_positions] = eos_weight

        # Compute weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        # Apply weights and mask padding
        mask = (shift_labels != -100).float()
        loss = (loss * loss_weights * mask).sum() / mask.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += 1

        # Calculate EM from logits (no generation needed - much faster!)
        with torch.no_grad():
            # Get predicted tokens from logits
            predictions = torch.argmax(logits, dim=-1)

            # logits[i] predicts token at position i+1, so shift predictions
            # predictions[:-1] corresponds to inputs['input_ids'][1:]
            answer_start = prompt_len
            answer_end = (labels[0] != -100).nonzero()[-1].item() + 1 if (labels[0] != -100).any() else answer_start

            # Extract answer tokens (accounting for the shift)
            pred_answer = predictions[0, answer_start-1:answer_end-1]
            true_answer = inputs['input_ids'][0, answer_start:answer_end]

            # Check exact match
            if len(pred_answer) == len(true_answer) and torch.equal(pred_answer, true_answer):
                exact_matches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'EM': f'{exact_matches}/{total_samples}'})

    return total_loss / total_samples, exact_matches, total_samples


def evaluate(model, tokenizer, loader, device):
    """Evaluate on validation subset."""
    model.eval()
    exact_matches = 0
    total_samples = 0
    failed_samples = []

    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            question = batch['question'][0]
            answer = batch['answer'][0]

            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            # Calculate dynamic max_new_tokens (estimate: answer length + buffer)
            estimated_answer_tokens = len(tokenizer(answer, return_tensors="pt")['input_ids'][0])
            max_new_tokens = min(estimated_answer_tokens + 20, 200)  # Cap at 200

            # Generate
            generated = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            pred_text = tokenizer.decode(generated[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

            # Check exact match
            is_match = calculate_exact_match(pred_text, answer)
            if is_match:
                exact_matches += 1
            else:
                failed_samples.append({
                    'prompt': prompt,
                    'generated': pred_text,
                    'expected': answer
                })

            total_samples += 1
            pbar.set_postfix({'EM': f'{exact_matches}/{total_samples}'})

    return exact_matches, total_samples, failed_samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, tokenizer = get_model()

    # Progressive expansion settings
    current_samples = 1000
    expansion_increment = 200
    val_em_threshold = 95.0    # Val EM must be >= 95% to expand
    expansion_history = []
    capacity_limit_found = False
    capacity_limit = None

    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Load checkpoint if exists
    checkpoint_path = "checkpoints/best.pth"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    start_epoch = 0
    best_em = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_em = checkpoint.get('best_em', 0)
        current_samples = checkpoint.get('current_samples', 1000)
        expansion_history = checkpoint.get('expansion_history', [])
        capacity_limit_found = checkpoint.get('capacity_limit_found', False)
        capacity_limit = checkpoint.get('capacity_limit', None)
        print(f"Resumed from epoch {start_epoch}, best EM: {best_em}")
        print(f"Current samples: {current_samples}\n")

    # Training loop
    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Dataset size: {current_samples} samples (indices: 0-{current_samples-1})")
        print(f"{'='*60}")

        # Load dataset with current sample size
        dataset = MMLUDataset("data/mmlu_combined.jsonl", max_samples=current_samples)
        train_loader = get_dataloader("data/mmlu_combined.jsonl", shuffle=True, max_samples=current_samples)

        # Train
        train_loss, train_em, train_total = train_epoch(model, tokenizer, train_loader, optimizer, device, eos_weight=3.0)
        train_em_percent = (train_em / train_total) * 100
        print(f"\nTrain Loss: {train_loss:.4f} | Train EM: {train_em}/{train_total} ({train_em_percent:.2f}%)")

        # Validation on random 10% subset (always randomized from current training data)
        val_size = int(0.1 * len(dataset))
        val_indices = random.sample(range(len(dataset)), val_size)
        val_subset = Subset(dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False)

        val_em, val_total, failed = evaluate(model, tokenizer, val_loader, device)
        val_em_percent = (val_em / val_total) * 100
        print(f"Val EM: {val_em}/{val_total} ({val_em_percent:.2f}%)")

        # Print failed samples
        if failed:
            print(f"\nFailed samples ({len(failed)}):")
            for i, sample in enumerate(failed[:5]):  # Show first 5
                print(f"\n{i+1}. Prompt: {sample['prompt']}")
                print(f"   Generated: {sample['generated']}")
                print(f"   Expected: {sample['expected']}")

        # Save checkpoint if best
        if val_em > best_em:
            best_em = val_em
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_em': best_em,
                'current_samples': current_samples,
                'expansion_history': expansion_history,
                'capacity_limit_found': capacity_limit_found,
                'capacity_limit': capacity_limit
            }, checkpoint_path)
            print(f"\n✓ Saved best checkpoint with EM: {best_em}/{val_total}")

        # Check for dataset expansion (only val EM >= 95%)
        if not capacity_limit_found and val_em_percent >= val_em_threshold:
            # Record current state before expansion
            expansion_record = {
                'samples': current_samples,
                'indices': f"0-{current_samples-1}",
                'epoch': epoch + 1,
                'train_em': train_em,
                'train_total': train_total,
                'train_em_percent': train_em_percent,
                'val_em': val_em,
                'val_total': val_total,
                'val_em_percent': val_em_percent,
                'val_failed_count': len(failed),
                'val_failed_samples': failed[:10],  # Store first 10 failed samples
                'expanded': True
            }
            expansion_history.append(expansion_record)

            # Expand dataset
            previous_samples = current_samples
            current_samples += expansion_increment

            print(f"\n{'='*60}")
            print(f"🚀 EXPANSION TRIGGERED!")
            print(f"Val EM: {val_em_percent:.2f}% >= {val_em_threshold}%")
            print(f"Expanding dataset: {previous_samples} → {current_samples} samples")
            print(f"New indices: 0-{current_samples-1}")
            print(f"{'='*60}\n")

        elif not capacity_limit_found and val_em_percent < val_em_threshold:
            # Check if we just expanded and performance dropped
            if expansion_history and expansion_history[-1]['samples'] < current_samples:
                capacity_limit_found = True
                capacity_limit = expansion_history[-1]['samples']

                print(f"\n{'='*60}")
                print(f"⚠️  CAPACITY LIMIT DETECTED!")
                print(f"Val EM: {val_em_percent:.2f}% (threshold: {val_em_threshold}%)")
                print(f"Capacity limit: {capacity_limit} samples")
                print(f"Reverting to previous dataset size")
                print(f"{'='*60}\n")

                # Revert to capacity limit
                current_samples = capacity_limit

        # Save expansion history to JSON
        history_data = {
            'expansion_history': expansion_history,
            'capacity_limit_found': capacity_limit_found,
            'capacity_limit': capacity_limit,
            'current_samples': current_samples,
            'final_indices': f"0-{current_samples-1}"
        }

        with open('logs/expansion_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)


if __name__ == "__main__":
    main()

