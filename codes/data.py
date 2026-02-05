"""MMLU Dataset loader."""

import json
from torch.utils.data import Dataset, DataLoader


class MMLUDataset(Dataset):
    """MMLU dataset from JSONL file."""

    def __init__(self, jsonl_path="data/mmlu_combined.jsonl", max_samples=1000):
        """Load MMLU dataset.

        Args:
            jsonl_path: Path to JSONL file
            max_samples: Maximum number of samples to load (default: 1000)
        """
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(jsonl_path="data/mmlu_combined.jsonl", shuffle=True, max_samples=3000):
    """Get MMLU dataloader with batch size 1.

    Args:
        jsonl_path: Path to JSONL file
        shuffle: Whether to shuffle data
        max_samples: Maximum number of samples to load (default: 3000)

    Returns:
        DataLoader with batch size 1
    """
    dataset = MMLUDataset(jsonl_path, max_samples=max_samples)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle)


if __name__ == "__main__":
    # Test the dataloader
    loader = get_dataloader()
    print(f"Dataset size: {len(loader.dataset)}")

    # Print first 3 examples
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        print(f"\nExample {i+1}:")
        print(f"  Question: {batch['question'][0]}")
        print(f"  Answer: {batch['answer'][0]}")
