import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


raw_datasets       = load_dataset("imdb")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
)

test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=2, collate_fn=data_collator
)


# for batch in train_dataloader:
#     break

# # t = {k: v.shape for k, v in batch.items()}
# # print(t)
