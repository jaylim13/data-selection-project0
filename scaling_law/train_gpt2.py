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
def heuristic_filter(example):
    perplexity = compute_perplexity(example["text"])
    # perplexities.append(perplexity)
    return 50 < perplexity <= 100  # Adjust thresholds based on your needs


"""Part 4 Tokenize"""


def tokenize_func(dataset, tokenizer, token_max=1_000_000, heuristic=False):
    """Tokenizer function for streamed dataset"""
    current_token_count = 0
    for example in dataset:
        if heuristic:
            if not heuristic_filter(example):
                continue
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


# tokenized_dataset = small_dataset.map(tokenize_func, batched=True).with_format("torch")
small_dataset = Dataset.from_generator(
    lambda: tokenize_func(randomized_dataset, tokenizer)
)

print(small_dataset)
total_samples = len(small_dataset)
print(total_samples)

model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

"""Part 5 Train GPT2"""

os.environ["WANDB_DISABLED"] = "true"

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
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
    train_dataset=small_dataset,
    data_collator=data_collator,
)

trainer.train()
