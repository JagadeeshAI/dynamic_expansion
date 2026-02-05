#!/usr/bin/env python3
"""Create sentence triplets from TinyStories dataset."""

import json
import re
from typing import List, Set


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving order."""
    # Split on . ! ? followed by optional quote and space/newline
    # This handles dialogue: "Hello." "Hi there."
    sentences = re.split(r'[.!?]+["\']?\s+', text)
    # Clean and filter empty sentences
    return [s.strip().strip('"\'') for s in sentences if s.strip()]


def create_triplets(input_file: str, output_file: str):
    """Create unique triplets from input JSONL file."""
    seen_pairs: Set[str] = set()
    triplet_id = 0
    max_triplets = 100

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if triplet_id >= max_triplets:
                break

            story = json.loads(line)
            text = story['text']

            # Split into sentences for THIS story only
            sentences = split_sentences(text)

            # Create triplets with sliding window WITHIN same story
            for i in range(len(sentences) - 2):
                if triplet_id >= max_triplets:
                    break

                s1, s2, s3 = sentences[i], sentences[i+1], sentences[i+2]

                # Clean sentences - remove newlines and extra whitespace
                s1 = ' '.join(s1.split())
                s2 = ' '.join(s2.split())
                s3 = ' '.join(s3.split())

                # Create unique key from first two sentences
                pair_key = s1 + "|||" + s2

                # Skip if we've seen this (s1, s2) pair before
                if pair_key in seen_pairs:
                    continue

                # Mark this pair as seen
                seen_pairs.add(pair_key)

                # Write triplet to output
                triplet = {
                    "s1": s1,
                    "s2": s2,
                    "s3": s3,
                    "id": triplet_id
                }
                f_out.write(json.dumps(triplet) + "\n")
                triplet_id += 1

    print(f"Created {triplet_id} unique triplets")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    create_triplets("data/tinystories_train.jsonl", "data/data.jsonl")
