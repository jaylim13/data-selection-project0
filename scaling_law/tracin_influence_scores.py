"""Gets all influence scores from the gradients"""

import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from datasets import load_dataset, Dataset, load_from_disk
import os

import matplotlib.pyplot as plt
import statistics
import numpy as np

from tracin_callback import TracinCallback, get_gradient_for_input, compute_influence

import json

checkpoint = "gpt2-finetuned/random-1e5"
model = AutoModelForCausalLM.from_pretrained(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = load_from_disk(
    "datasets/train_dataset_randomized_1e5"
)  # Load the dataset from disk
checkpoint_grads = torch.load("datasets/tracin_grads_1e5.pt")
print(len(checkpoint_grads))
# print(checkpoint_grads)


def get_all_influence_scores(
    model, dataset, checkpoint_grads, tokenizer, device, batch_size=1
):
    """
    Computes TracIn influence scores for each example in the dataset using 1 GPU.
    Assumes batch_size = 1 for accurate attribution.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    scores = []

    for batch in tqdm.tqdm(dataloader):
        # Tokenize (if not already tokenized)
        if isinstance(batch, dict):
            inputs = {k: v.to(device) for k, v in batch.items()}
        else:
            inputs = tokenizer(
                batch["text"], return_tensors="pt", truncation=True, padding=True
            ).to(device)

        # Compute gradient for this input
        target_grad = get_gradient_for_input(model, inputs, device)

        # Compute influence with each checkpoint
        sample_scores = compute_influence(checkpoint_grads, target_grad)

        # You can average across checkpoints or keep all
        scores.append(np.mean(sample_scores))

    return scores


influence_scores = get_all_influence_scores(
    model, train_dataset, checkpoint_grads, tokenizer, device, batch_size=1
)

# scored_dataset = list(zip(influence_scores, train_dataset))

# print(influence_scores)

with open(
    f"datasets/influence_scores/influence_scores_list_{len(train_dataset)}.json", "w"
) as f:
    json.dump(influence_scores, f)

with open("influence_scores_list.json", "r") as f:
    influence_scores = json.load(f)

# Now `influence_scores` is a list that you can use
# print(influence_scores)  # F
