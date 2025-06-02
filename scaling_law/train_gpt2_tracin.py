"""Train model on random data, but extracts the gradients during training to later compute influence scores."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from datasets import load_dataset, Dataset
import os

import matplotlib.pyplot as plt
import statistics
import numpy as np

from tracin_callback import (
    TracinCallback,
    TracinTrainer,
    get_gradient_for_input,
    compute_influence,
)

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

dataset = load_dataset(
    "allenai/c4", "en", split="train", streaming=True, trust_remote_code=True
)

randomized_dataset = dataset.shuffle(seed=42)
# small_dataset = dataset.train_test_split(test_size=0.1)
print(randomized_dataset)

"""Part 4 Tokenize"""


def tokenize_train(dataset, tokenizer, token_max=100_000):
    """Tokenizer function for streamed dataset"""
    current_token_count = 0
    for example in dataset:
        # print(example)
        tokenized = tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=512
        )
        input_ids = tokenized["input_ids"]
        tokenized["labels"] = tokenized["input_ids"].copy()
        current_token_count += len(input_ids)
        if current_token_count >= token_max:
            break
        yield tokenized


train_dataset = Dataset.from_generator(
    lambda: tokenize_train(randomized_dataset, tokenizer)
)
train_dataset = train_dataset.with_format("torch")
train_dataset.save_to_disk("./datasets/train_dataset_randomized_1e5")

print(train_dataset)
total_samples = len(train_dataset)
print(total_samples)

model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

"""Part 5 Train GPT2"""

os.environ["WANDB_DISABLED"] = "true"

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned/tracin-random",
    run_name="project0",
    eval_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    fp16=False,
    local_rank=local_rank,
    dataloader_num_workers=6,
    learning_rate=1e-4,
    logging_steps=total_samples // (16 * 2 * 4),
    logging_dir="./logs",
    remove_unused_columns=False,
)

capture_callback = TracinCallback(capture_every_n_steps=1)

# Trainer
trainer = TracinTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[capture_callback],
)

trainer.train()
torch.save(capture_callback.checkpoint_grads, "datasets/tracin_grads_1e5.pt")

print(f"Number of checkpoints captured: {len(capture_callback.checkpoint_grads)}")

# Optional: Inspect shape and type of the first gradient vector
if capture_callback.checkpoint_grads:
    print("First gradient vector shape:", capture_callback.checkpoint_grads[0].shape)
    print("First few values:", capture_callback.checkpoint_grads[0][:5])
else:
    print("No gradients were captured.")
