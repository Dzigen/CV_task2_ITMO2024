import numpy as np
from tqdm import tqdm
import torch
import gc

def train(emb_model, cls_model, loader, optimizer, 
          scheduler, criterion, device):
    emb_model.train()
    cls_model.train()

    losses = []
    process = tqdm(loader)
    for batch in process:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs1 = emb_model(batch['images'])
        outputs2 = cls_model(outputs1, batch['labels'])
        loss = criterion(outputs2, batch['labels'])

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        process.set_postfix({"avg_loss": np.mean(losses)})
        gc.collect()
        torch.cuda.empty_cache()

    return losses