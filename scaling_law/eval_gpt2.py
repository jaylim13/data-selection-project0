import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from datasets import load_dataset, Dataset
import os
from transformers import DataCollatorForLanguageModeling


# Initialize the distributed process group for multi-GPU
print(torch.cuda.device_count())

# Load any model from checkpoint
checkpoint = "./gpt2-finetuned/perplexity-1e8"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer_checkpoint = AutoTokenizer.from_pretrained(checkpoint)
# torch.distributed.init_process_group(backend="nccl")
# local_rank = int(os.environ["LOCAL_RANK"])
# device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)
model.to(device)
model.eval()


# def run_query(query, model, tokenizer, max_new_tokens=50):
#     inputs = tokenizer(query, return_tensors="pt").to(device)
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.95,
#         )
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#     return generated_text


# query = "I am"
# response = run_query(query, model, tokenizer_checkpoint)
# print(f"GPT-2: {response}", flush=True)

# Evaluation Loss on Test Set
test_dataset = load_dataset(
    "allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True
)
test_dataset = test_dataset.shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.resize_token_embeddings(len(tokenizer))


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


def tokenize_func(dataset, tokenizer, token_max=1_000_000, heuristic=False):
    """Tokenizer function for streamed dataset"""
    current_token_count = 0
    for example in dataset:
        if heuristic:
            if not heuristic_filter(example):
                continue
        print(example)
        tokenized = tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=512
        )
        input_ids = tokenized["input_ids"]
        tokenized["labels"] = tokenized["input_ids"].copy()

        current_token_count += len(input_ids)
        if current_token_count >= token_max:
            break
        yield tokenized


test_dataset = Dataset.from_generator(lambda: tokenize_func(test_dataset, tokenizer))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

eval_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=4,
    do_eval=True,
    dataloader_num_workers=2,
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Run evaluation
eval_results = trainer.evaluate()
print(eval_results)
