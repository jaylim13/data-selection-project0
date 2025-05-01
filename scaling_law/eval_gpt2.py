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
checkpoint = "./gpt2-finetuned-old/checkpoint-195312"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer_checkpoint = AutoTokenizer.from_pretrained(checkpoint)
# torch.distributed.init_process_group(backend="nccl")
# local_rank = int(os.environ["LOCAL_RANK"])
# device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)
model.to(device)
model.eval()


def run_query(query, model, tokenizer, max_new_tokens=50):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


query = "I am"
response = run_query(query, model, tokenizer_checkpoint)
print(f"GPT-2: {response}", flush=True)

# Evaluation Loss on Test Set
test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_function(example):
    # Tokenize the example
    tokenized_example = tokenizer(example["text"], truncation=True, max_length=512)
    # Ensure non-empty input
    if len(tokenized_example["input_ids"]) > 0:
        return tokenized_example
    else:
        return None  # Skip empty sequences


# Filter out None values (empty sequences)
tokenized_test = test_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
).filter(lambda example: example is not None)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_checkpoint,
    mlm=False,  # Causal LM uses non-masked language modeling
)

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
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

# Run evaluation
eval_results = trainer.evaluate()
print(eval_results)
