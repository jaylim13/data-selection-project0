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

"""Part 2 Load gpt2"""
# load the gpt-2 model and respective tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to MPS (if your computer is m1 macbook like mine) or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


def tokenize_func(dataset, tokenizer, token_max=100_000):
    """Tokenizer function for streamed dataset"""
    current_token_count = 0
    for example in dataset:
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
    per_device_train_batch_size=1,
    max_steps=total_samples,
    learning_rate=1e-4,
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
