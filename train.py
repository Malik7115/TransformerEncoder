
import os
import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import DataLoader
import einops
from transformerModel import EncoderStack
from cDataloader import tokenized_datasets, data_collator
from torchmetrics.functional import accuracy


###################### Params ######################

bs          = 64
epochs      = 200
device      = 'cuda'
lr          = 1e-5
seq_len     = 512
num_classes = 2
embedding_dim = 256
heads = 6
layers = 3

####################################################



train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=bs, collate_fn=data_collator, drop_last=True
)

test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=bs//2, collate_fn=data_collator, drop_last=True
)





model = EncoderStack(dim=embedding_dim, heads=heads, layers=layers)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


if __name__ == "__main__":

    os.system('clear')
    print("Training Started")
    
    best_acc = 0
    for epoch in range(epochs):
        running_acc = 0 
        t_steps = 0

        for steps, batch in enumerate(train_dataloader):
            t_steps += 1
            optimizer.zero_grad()
            labels     = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            out    = model(input_ids, mask=attention_mask)
            loss   = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_acc = accuracy(torch.argmax(out, dim = 1), labels)
            running_acc += train_acc.float() 

        running_acc = running_acc/t_steps
        # np.save('/home/jarvis/Projects/transformer/ckpts' + str(epoch) + '.npy', gen)
        print('Epoch: ', epoch,'\tLoss Train: ', loss.item(), '\tAcc Train: ', running_acc)
    
        if(running_acc > best_acc):
            best_acc = running_acc
            torch.save(model.state_dict(), '/home/jarvis/Projects/transformer/ckpts/model_' + str(epoch) + '.pth')