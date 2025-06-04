# sample.py
import torch
import matplotlib.pyplot as plt
from model import VDM, sigma2, variance_preserving_map
from utils import rawarrview, reshape_image_batch

def generate(model, device, shape, conditioning, guidance_weight=0.0):
    """
    Generate samples from the model.
    
    Args:
      model: an instance of VDM.
      shape: tuple giving the batch shape (e.g. (B,)).
      conditioning: tensor of condition indices.
      guidance_weight: weight for classifier guidance.
      
    Returns:
      Logits from the decoder (apply sigmoid for probabilities).
    """
    # Generate latent starting from standard normal.
    z_t = torch.randn(*shape, model.embedding_dim, device=device)
    T = model.timesteps
    for i in range(T):
        z_t = model.sample_step(i, T, z_t, conditioning, guidance_weight)
    g0 = model.gamma_fn(torch.tensor(0., device=device))
    var0 = sigma2(g0)
    z0_rescaled = z_t / torch.sqrt(1. - var0)
    logits = model.decoder(z0_rescaled, model.embedding_vectors(conditioning))
    return logits

def recon(model, device, t, ims, conditioning):
    """
    Reconstruct images from corrupted latents.
    
    Args:
      t: a time scalar (e.g. 0.8) as a torch.tensor.
      ims: input images.
      conditioning: labels.
      
    Returns:
      Logits from the decoder.
    """
    z_0 = model.encode(ims, conditioning)
    T = model.timesteps
    tn = torch.ceil(t * T)
    t_val = tn / T
    g_t = model.gamma_fn(t_val)
    eps = torch.randn_like(z_0)
    if g_t.dim() == 0:
        g_t = g_t.expand(z_0.size(0))
    z_t = variance_preserving_map(z_0, g_t.unsqueeze(1), eps)
    # Run reverse steps from step (T - tn) to T.
    for i in range(int(T - tn.item()), T):
            z_t = model.sample_step(i, T, z_t, conditioning)
    g0 = model.gamma_fn(torch.tensor(0., device=device))
    var0 = sigma2(g0)
    z0_rescaled = z_t / torch.sqrt(1. - var0)
    logits = model.decoder(z0_rescaled, conditioning)
    return logits

def viewer(x, cmap='bone_r'):
    """
    Display a grid of images. Expects x as a NumPy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    rawarrview(reshape_image_batch(x), cmap=cmap)
