#!/usr/bin/env python
"""
train.py - Contains the training loop and evaluation routines.
Supports both:
  • Discrete‑time VDM (model(images, conditioning) -> loss)
  • Continuous‑time VDM (model((images, conditioning)) -> (loss, metrics))
Continuous‑time inputs are permuted from [B,H,W,C] to [B,C,H,W].
This code is otherwise unchanged from your original notebook logic.
"""

import math
import os
import torch
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
from utils import param_count

def loss_fn(model, images, conditioning, beta=1.0):
    """
    Computes the loss in bits-per-dimension.
    Dispatches to the correct forward signature:
      - Continuous‑time: forward((x, cond)) returns (loss, metrics)
      - Discrete‑time: forward(images, conditioning) returns loss
    """
    if hasattr(model, "cfg"):
        # Continuous‑time expects [B,C,H,W]
        x = images.permute(0, 3, 1, 2)
        loss, _ = model((x, conditioning))
    else:
        # Discrete‑time uses original signature
        loss = model(images, conditioning)

    prod_dims = images[0].numel()  # assuming images shape [B, 28, 28, 1]
    rescale_to_bpd = 1.0 / (prod_dims * math.log(2.0))
    return loss * rescale_to_bpd

def train_model(model, train_loader, test_loader, args, device):
    TSTEPS = args.train_steps
    learning_rate = args.learning_rate

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=1e-4
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TSTEPS, eta_min=1e-5
        )
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=TSTEPS
        )
    else:
        raise ValueError("Unsupported scheduler type")

    loss_history = []
    model.train()

    train_iter = iter(train_loader)
    for step in trange(TSTEPS):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device)
        conditioning = labels.to(device)

        # Discrete‑time original code uses zeros for conditioning
        if not hasattr(model, "cfg"):
            conditioning = torch.zeros(
                images.size(0), dtype=torch.long, device=device
            )

        optimizer.zero_grad()
        loss_val = loss_fn(model, images, conditioning, beta=1.0)
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss_val.item())

        if step % 100 == 0:
            print(f"Step {step}: loss {loss_val.item():.4f}")

    # Save the loss curve instead of displaying it.
    if len(loss_history) > 1000:
        os.makedirs("model", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history[1000:])
        plt.xlabel("Iteration")
        plt.ylabel("Loss (bpd)")
        plt.title("Training Loss")
        loss_curve_path = os.path.join("model", "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"Loss curve saved to '{loss_curve_path}'")

    print(f"Total parameters: {param_count(model):,}")

def evaluate_model(model, test_loader, args, device):
    """
    Evaluates the model on one batch.
    Continuous‑time: prints all returned metrics.
    Discrete‑time: prints total loss only.
    """
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        conditioning = labels.to(device)

        if not hasattr(model, "cfg"):
            conditioning = torch.zeros(
                images.size(0), dtype=torch.long, device=device
            )

        if hasattr(model, "cfg"):
            # Continuous‑time: permute and call forward((x, cond))
            x = images.permute(0, 3, 1, 2)
            loss, metrics = model((x, conditioning))
            print("Evaluation metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            # Discrete‑time: original evaluation logic
            loss = model(images, conditioning)
            print(f"Evaluation loss: {loss.item():.4f}")
