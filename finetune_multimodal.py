#!/usr/bin/env python3
"""
finetune_multimodal.py

Fine-tune the MultimodalFusionModel on synthetic AI2-THOR dataset (JSONL format).

Supports all modes: 'action', 'description', 'vqa', 'progress'
with automatic evaluation and logging.
"""

import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from multimodal_network import MultimodalFusionModel  # ← your model file

from transformers import AutoTokenizer
from datasets import load_dataset

# ================================================================
# CONFIGURATION
# ================================================================
DATA_PATH = "./synthetic_dataset/data.jsonl"
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 10
VAL_SPLIT = 0.1
LOG_INTERVAL = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# DATASET CLASS
# ================================================================

class MultimodalThorDataset(Dataset):
    def __init__(self, data, tokenizer, mode_filter=None):
        self.data = [x for x in data if (mode_filter is None or x["mode"] == mode_filter)]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        mode = ex["mode"]
        img = Image.open(ex["frame_path"]).convert("RGB")
        prompt = ex["prompt"]
        target = None

        if mode == "action":
            target = torch.tensor(ex["action_id"]).long()
        elif mode in ("description", "vqa"):
            target_text = ex.get("target_text", "")
            target = self.tokenizer(
                target_text, return_tensors="pt", padding="max_length",
                truncation=True, max_length=64
            ).input_ids.squeeze(0)
        elif mode == "progress":
            target = torch.tensor(ex.get("progress_value", 0.0)).float()

        return {
            "image": img,
            "prompt": prompt,
            "target": target,
            "mode": mode
        }


def collate_fn(batch):
    """Group by mode dynamically for flexible batching."""
    # You could separate modes, but for simplicity here we do same-batch mode only
    modes = [b["mode"] for b in batch]
    assert len(set(modes)) == 1, "Batch must contain only one mode type!"
    mode = modes[0]

    images = [b["image"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]

    if isinstance(targets[0], torch.Tensor):
        if targets[0].dim() == 0:
            targets = torch.stack(targets)
        else:
            targets = torch.nn.utils.rnn.pad_sequence(
                targets, batch_first=True, padding_value=0
            )

    return {"images": images, "prompts": prompts, "targets": targets, "mode": mode}


# ================================================================
# TRAINING UTILITIES
# ================================================================

def train_one_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc=f"🧩 Training epoch {epoch}")):
        images = batch["images"]
        prompts = batch["prompts"]
        targets = batch["targets"].to(device)
        mode = batch["mode"]

        outputs = model(images, prompts, targets=targets, mode=mode)
        loss = outputs.get("loss", None)
        if loss is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % LOG_INTERVAL == 0:
            avg_loss = total_loss / (step + 1)
            print(f"[Epoch {epoch} | Step {step+1}] Mode={mode} Loss={avg_loss:.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🧪 Evaluating"):
            images = batch["images"]
            prompts = batch["prompts"]
            targets = batch["targets"].to(device)
            mode = batch["mode"]

            outputs = model(images, prompts, targets=targets, mode=mode)
            loss = outputs.get("loss", None)
            if loss is not None:
                total_loss += loss.item()
    return total_loss / len(dataloader)


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    print("🚀 Loading dataset...")
    raw_ds = load_dataset("json", data_files=DATA_PATH)["train"]
    data = [dict(x) for x in raw_ds]

    # Split train/val
    random.shuffle(data)
    split_idx = int(len(data) * (1 - VAL_SPLIT))
    train_data, val_data = data[:split_idx], data[split_idx:]

    # Build tokenizer from model
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets (you can filter mode here for specialized runs)
    train_ds = MultimodalThorDataset(train_data, tokenizer)
    val_ds = MultimodalThorDataset(val_data, tokenizer)

    # For simplicity, train only one mode per dataloader iteration
    # (you can loop over modes later)
    modes = ["action", "vqa", "description", "progress"]

    # Model
    model = MultimodalFusionModel().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Logging
    print(f"Training on {len(train_data)} samples; validation on {len(val_data)}")

    for epoch in range(1, EPOCHS + 1):
        epoch_losses = []
        for mode in modes:
            print(f"\n=== Mode: {mode.upper()} ===")
            train_loader = DataLoader(
                [x for x in train_ds if x["mode"] == mode],
                batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
            )
            val_loader = DataLoader(
                [x for x in val_ds if x["mode"] == mode],
                batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
            )

            train_loss = train_one_epoch(model, train_loader, optimizer, epoch, DEVICE)
            val_loss = evaluate(model, val_loader, DEVICE)

            print(f"📉 Epoch {epoch} | Mode {mode} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            epoch_losses.append((mode, train_loss, val_loss))

        # Save checkpoint
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"./checkpoints/mmfusion_epoch{epoch}.pt")

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
