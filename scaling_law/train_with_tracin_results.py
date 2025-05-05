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


train_dataset = load_from_disk("datasets/train_dataset_randomized_1e5")
with open(
    f"datasets/influence_scores/influence_scores_list_{len(train_dataset)}.json", "r"
) as f:
    influence_scores = json.load(f)

scored_dataset = list(zip(influence_scores, train_dataset))

sorted_scored = sorted(scored_dataset, key=lambda x: x[0], reverse=True)
print(len(influence_scores))
print(len(train_dataset))
print(len(sorted_scored))

# top_k_dataset = []

os.environ["HF_HUB_READ_TIMEOUT"] = "60"
os.environ["HF_HUB_CONNECT_TIMEOUT"] = "60"

"""Part 2 Load gpt2"""
# load the gpt-2 model and respective tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
# Initialize the distributed process group for multi-GPU
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

print(device)
model.to(device)

print(torch.cuda.device_count())


"""Part 4 Tokenize"""


def tokenize_train(tokenizer, token_max=50_000):
    """Tokenizer function for streamed dataset"""
    top_k = []
    current_token_count = 0
    for score, example in sorted_scored:
        # tokenized = tokenizer(
        #     example["text"], truncation=True, padding="max_length", max_length=512
        # )
        # input_ids = tokenized["input_ids"]
        # tokenized["labels"] = tokenized["input_ids"].copy()
        current_token_count += len(example["input_ids"])
        if current_token_count >= token_max:
            break
        top_k.append(example)

    return top_k


tokenized_top_k = tokenize_train(tokenizer)
tokenized_top_k = Dataset.from_list(tokenized_top_k)

print(tokenized_top_k)
total_samples = len(tokenized_top_k)
print(total_samples)

model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

"""Part 5 Train GPT2"""

os.environ["WANDB_DISABLED"] = "true"

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned/pruned-outliers",
    run_name="project0",
    eval_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    fp16=True,
    local_rank=local_rank,
    dataloader_num_workers=6,
    learning_rate=1e-4,
    logging_dir="./logs",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_top_k,
    data_collator=data_collator,
)

trainer.train()

# print("Sample size: " + str(len(perplexities)))
# print("mean: " + str(statistics.mean(perplexities)))
# print("lower quartile: " + str(np.percentile(perplexities, 25)))
# print("median: " + str(statistics.median(perplexities)))
# print("upper quartile: " + str(np.percentile(perplexities, 75)))
