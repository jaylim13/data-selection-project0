import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset, Dataset
import os

import matplotlib.pyplot as plt
import statistics
import numpy as np

os.environ["HF_HUB_READ_TIMEOUT"] = "60"
os.environ["HF_HUB_CONNECT_TIMEOUT"] = "60"

"""Part 2 Load gpt2"""
# load the gpt-2 model and respective tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
perplexities = []

# Initialize the distributed process group for multi-GPU
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

print(device)
model.to(device)

print(torch.cuda.device_count())


"""Part 3 Load Dataset"""
# load small sample of common crawl dataset
# if int(os.environ.get("LOCAL_RANK", 0)) == 0:
#     print("Main process loading the streaming dataset...")
dataset = load_dataset(
    "allenai/c4", "en", split="train", streaming=True, trust_remote_code=True
)
# rank = int(os.environ.get("LOCAL_RANK", 0))
# world_size = int(os.environ.get("WORLD_SIZE", 1))
# dataset = dataset.shard(num_shards=world_size, index=rank)

randomized_dataset = dataset.shuffle(seed=42)
# small_dataset = dataset.train_test_split(test_size=0.1)
print(randomized_dataset)

"""Part 3.5 Perplexity Heuristic"""


def compute_perplexity(text):
    """Compute perplexity of a given text using GPT-2."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    with torch.no_grad():  # Disable gradient computation for efficiency
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()  # Convert loss to perplexity


# Define heuristic filter: Keep texts with moderate perplexity
def heuristic_filter(example, low_perp_full, med_perp_full, high_perp_full):
    perplexity = compute_perplexity(example["text"])
    perplexities.append(perplexity)
    # perplexities.append(perplexity)
    if perplexity < 22 and not low_perp_full:
        return "low"
    elif 22 <= perplexity < 55 and not med_perp_full:
        return "medium"
    elif 55 <= perplexity <= 400 and not high_perp_full:
        return "high"
    else:
        return None


# def compute_perplexity(text):
#     """Compute perplexity of a given text using GPT-2."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
#         device
#     )
#     with torch.no_grad():  # Disable gradient computation for efficiency
#         loss = model(**inputs, labels=inputs["input_ids"]).loss
#     return torch.exp(loss).item()  # Convert loss to perplexity


# # Define heuristic filter: Keep texts with moderate perplexity
# def heuristic_filter(example):
#     perplexity = compute_perplexity(example["text"])
#     # perplexities.append(perplexity)
#     return 25 < perplexity <= 55  # Adjust thresholds based on your needs


"""Part 4 Tokenize"""


def tokenize_train(dataset, tokenizer, token_max=100_000_000, heuristic=True):
    """Tokenizer function for streamed dataset"""
    count_dict = {"low": 0, "medium": 0, "high": 0}
    current_token_count = 0
    target_perp = None
    for example in dataset:
        perplexities.append(compute_perplexity(example["text"]))
        # if lb == 0 and ub == 50 and current_token_count > 0.1 * token_max:
        #     lb = 50
        #     ub = 100
        # elif lb == 50 and ub == 100 and current_token_count > 0.9 * token_max:
        #     lb = 100
        #     ub = 1000
        if heuristic:
            target_perp = heuristic_filter(
                example,
                count_dict["low"] > 0.05 * token_max,
                count_dict["medium"] > 0.8 * token_max,
                count_dict["high"] > 0.15 * token_max,
            )
            if not target_perp:
                continue

        # print(example)
        tokenized = tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=512
        )
        input_ids = tokenized["input_ids"]
        tokenized["labels"] = tokenized["input_ids"].copy()
        if target_perp:
            count_dict[target_perp] += len(input_ids)
        current_token_count += len(input_ids)
        if current_token_count >= token_max:
            break
        yield tokenized


train_dataset = Dataset.from_generator(
    lambda: tokenize_train(randomized_dataset, tokenizer)
)
train_dataset = train_dataset.with_format("torch")

print(train_dataset)
total_samples = len(train_dataset)
print(total_samples)

model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
# train_dataset = train_dataset.shuffle(seed=42)

# def tokenize_func(dataset, tokenizer, token_max=100_000, heuristic=True):
#     """Tokenizer function for streamed dataset"""
#     current_token_count = 0
#     for example in dataset:
#         if heuristic:
#             if not heuristic_filter(example):
#                 continue
#         # print(example)
#         tokenized = tokenizer(
#             example["text"], truncation=True, padding="max_length", max_length=512
#         )
#         input_ids = tokenized["input_ids"]
#         tokenized["labels"] = tokenized["input_ids"].copy()

#         current_token_count += len(input_ids)
#         if current_token_count >= token_max:
#             break
#         yield tokenized


# tokenized_dataset = small_dataset.map(tokenize_func, batched=True).with_format("torch")
# small_dataset = Dataset.from_generator(
#     lambda: tokenize_train(randomized_dataset, tokenizer)
# )

# print(small_dataset)
# total_samples = len(small_dataset)
# print(total_samples)

# model.resize_token_embeddings(len(tokenizer))
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

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
    logging_steps=total_samples // (16 * 2 * 4),
    logging_dir="./logs",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# print("Sample size: " + str(len(perplexities)))
# print("mean: " + str(statistics.mean(perplexities)))
# print("lower quartile: " + str(np.percentile(perplexities, 25)))
# print("median: " + str(statistics.median(perplexities)))
# print("upper quartile: " + str(np.percentile(perplexities, 75)))
