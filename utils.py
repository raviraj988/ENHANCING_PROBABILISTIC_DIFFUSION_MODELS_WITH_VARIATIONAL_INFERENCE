#!/usr/bin/env python
"""
utils.py - Utility functions for image processing and model analysis.
Contains functions such as imify, rawarrview, reshape_image_batch, zoom, imgviewer,
viewer, and param_count.
Do NOT modify any of the original code.
"""

import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from IPython.display import display_png

def imify(arr, vmin=None, vmax=None, cmap=None, origin=None):
    """Convert an array to an image using a matplotlib colormap."""
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    if origin is None:
        origin = mpl.rcParams["image.origin"]
    if origin == "lower":
        arr = arr[::-1]
    rgba = sm.to_rgba(arr, bytes=True)
    return rgba

def rawarrview(array, **kwargs):
    """Visualize an array as an image in notebooks."""
    f = io.BytesIO()
    imarray = imify(array, **kwargs)
    plt.imsave(f, imarray, format="png")
    f.seek(0)
    dat = f.read()
    f.close()
    display_png(dat, raw=True)

def reshape_image_batch(array, cut=None, rows=None, axis=0):
    """Reshape an array of shape [n, x, y, ...] into a grid for visualization."""
    original_shape = array.shape
    assert len(original_shape) >= 2, "array must be at least 3-dimensional."
    if cut is None:
        cut = original_shape[axis]
    if rows is None:
        rows = int(math.sqrt(cut))
    cols = cut // rows
    cut = cols * rows
    leading = original_shape[:axis]
    x_width = original_shape[axis + 1]
    y_width = original_shape[axis + 2]
    remaining = original_shape[axis + 3:]
    array = array[:cut]
    array = array.reshape(leading + (rows, cols, x_width, y_width) + remaining)
    array = np.moveaxis(array, axis + 2, axis + 1)
    array = array.reshape(leading + (rows * x_width, cols * y_width) + remaining)
    return array

def zoom(im, k, axes=(0, 1)):
    """Zoom an image by repeating pixels k times on given axes."""
    for ax in axes:
        im = np.repeat(im, k, axis=ax)
    return im

def imgviewer(im, zoom_factor=3, cmap='bone_r', normalize=False, **kwargs):
    """Display an image with optional zoom and normalization."""
    if normalize:
        im = im - im.min()
        im = im / im.max()
    return rawarrview(zoom(im, zoom_factor), cmap=cmap, **kwargs)

def viewer(x, **kwargs):
    """Wrapper to visualize images; converts tensor to numpy if needed."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return rawarrview(reshape_image_batch(x), **kwargs)

def param_count(model):
    """Return the total number of parameters in a torch.nn.Module."""
    return sum(p.numel() for p in model.parameters())
