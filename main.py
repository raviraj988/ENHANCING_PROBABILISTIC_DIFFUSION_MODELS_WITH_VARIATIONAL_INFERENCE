#!/usr/bin/env python
"""
main.py - Entry point for VDM Diffusion Model.
Supports two operation modes:
  1. "train": Train the model (discrete or continuous time), evaluate it, and save a checkpoint.
  2. "generate": Load a saved model checkpoint and generate images without retraining.
You can choose the dataset (MNIST, FashionMNIST, EMNIST, KMNIST, or QMNIST), discrete vs continuous time,
and noise schedule type. Generated images are saved into "images/" and checkpoints into "model/".
"""

import argparse
import os
import torch
from data import load_data
from model import get_model, generate
from train import train_model, evaluate_model
import utils
from torch import argmax
import math
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="VDM Diffusion Model Training & Generation")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for the dataset")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "QMNIST"],
                        default="MNIST", help="Dataset to use")
    parser.add_argument("--train_steps", type=int, default=20000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for training and evaluation")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear"],
                        help="Learning rate scheduler type")
    parser.add_argument("--learning_rate", type=float, default=8e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--continuous_time", action="store_true",
                        help="Use continuous time model (jointly trains noise schedule)")
    parser.add_argument("--noise_schedule", type=str, choices=["fixed", "learned"], default="fixed",
                        help="For continuous time: 'fixed' or 'learned'")
    parser.add_argument("--mode", type=str, choices=["train", "generate"], default="train",
                        help="Operation mode: train or generate")
    parser.add_argument("--gen_mode", type=str, default="conditional", choices=["conditional", "unconditional"],
                        help="Generation mode")
    parser.add_argument("--num_samples", type=int, default=128,
                        help="Number of images to generate")
    parser.add_argument("--guidance_weight", type=float, default=0.0,
                        help="Guidance weight for generation")
    parser.add_argument("--sample_steps", type=int, default=100,
                        help="Number of steps for continuous-time sampling")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Ensure directories exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # Define checkpoint paths for discrete and continuous time models
    discrete_ckpt = os.path.join("model", "vdm_discrete.pth")
    cont_ckpt     = os.path.join("model", "vdm_continuous.pth")

    # TRAINING
    if args.mode == "train":
        # Load data
        train_loader, test_loader = load_data(args)

        # Build model (discrete or continuous)
        model = get_model(args).to(device)

        # Train
        train_model(model, train_loader, test_loader, args, device)

        # Save checkpoint
        if args.continuous_time:
            torch.save(model.state_dict(), cont_ckpt)
            print(f"Continuous‑time model saved to '{cont_ckpt}'")
        else:
            torch.save(model.state_dict(), discrete_ckpt)
            print(f"Discrete‑time model saved to '{discrete_ckpt}'")

        # Evaluate
        evaluate_model(model, test_loader, args, device)

    # GENERATION
    elif args.mode == "generate":
        # Instantiate model architecture
        model = get_model(args).to(device)

        # Load matching checkpoint (for continuous, use cont_ckpt; else discrete_ckpt)
        ckpt_to_load = cont_ckpt if args.continuous_time else discrete_ckpt
        if not os.path.exists(ckpt_to_load):
            print(f"No checkpoint found at '{ckpt_to_load}'. Please train first with matching --continuous_time flag.")
            return
        model.load_state_dict(torch.load(ckpt_to_load, map_location=device))
        print(f"Loaded {'continuous' if args.continuous_time else 'discrete'} checkpoint from '{ckpt_to_load}'")

        # Generate samples for discrete-time model or continuous-time model.
        # (Here we assume discrete-time generation; continuous-time sampling code is provided in the original file.)
        if args.gen_mode == "conditional":
            cond = torch.arange(args.num_samples, device=device) % 11
        else:
            cond = torch.zeros(args.num_samples, dtype=torch.long, device=device)

        if hasattr(model, "embedding_dim"):
            # Discrete-time model generation
            gen_dist = generate(model, (args.num_samples,), cond,
                                guidance_weight=args.guidance_weight, device=device)
        else:
            # Continuous-time generation (if applicable)
            from torch import linspace
            B = args.num_samples
            z = torch.randn((B, *model.image_shape), device=device)
            steps = torch.linspace(1.0, 0.0, args.sample_steps + 1, device=device)
            for i in range(args.sample_steps):
                z = model.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples=False)
            logprobs = model.log_probs_x_z0(z_0=z)
            x = argmax(logprobs, dim=-1)
            gen = x.float() / (model.vocab_size - 1)
            gen_dist = type("dummy", (object,), {"mean": gen})()  # Create dummy object to mimic gen_dist.mean

        samples = gen_dist.mean

        # Prepare grid for saving (use utils.reshape_image_batch)
        grid = utils.reshape_image_batch(samples.squeeze(-1).detach().cpu().numpy(),
                                         rows=int(math.sqrt(args.num_samples)))
        import matplotlib.pyplot as plt
        # Name generated file to include dataset name
        gen_filename = f"generated_{args.dataset}_{'cond' if args.gen_mode=='conditional' else 'uncond'}_gw{args.guidance_weight}.png"
        gen_path = os.path.join("images", gen_filename)

        # If grid is 2D (grayscale), expand dims to 3 channels for saving if needed.
        if grid.ndim == 2:
            grid = np.stack([grid]*3, axis=-1)
        elif grid.ndim == 3 and grid.shape[-1] == 1:
            grid = np.repeat(grid, 3, axis=-1)
        plt.imsave(gen_path, grid, cmap='bone_r')
        print(f"Sample generation saved to '{gen_path}'")

if __name__ == "__main__":
    main()
