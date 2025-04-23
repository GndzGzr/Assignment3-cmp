#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash/No-Flash Photography - Project Demo for Parts 1 & 2
This script demonstrates key aspects of bilateral filtering and gradient domain processing
from the Flash/No-Flash photography implementation.

For Jupyter notebook conversion:
- Code cells are demarcated with comments "# CODE CELL: <description>"
- Add markdown cells where indicated by "# MARKDOWN CELL: <description>"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from tqdm import tqdm

# Import our implementations
from classes_functions.bilateral_filter import BilateralFilter
from classes_functions.poisson_solver import PoissonSolver

# Helper utility for displaying images side by side
def display_images(images, titles=None, figsize=(15, 10), cmaps=None):
    """
    Display multiple images side by side
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size (width, height)
        cmaps: List of colormaps for each image
    """
    n = len(images)
    if cmaps is None:
        cmaps = ['viridis'] * n
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n)]
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        if len(img.shape) == 3 and img.shape[2] == 3:  # Color image
            axes[i].imshow(np.clip(img, 0, 1))
        else:  # Grayscale image
            axes[i].imshow(np.clip(img, 0, 1), cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to load flash/no-flash image pair
def load_image_pair(flash_path, noflash_path, normalize=True):
    """
    Load a flash/no-flash image pair
    
    Args:
        flash_path: Path to flash image
        noflash_path: Path to no-flash image
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Tuple of (flash_image, noflash_image)
    """
    flash = io.imread(flash_path)
    noflash = io.imread(noflash_path)
    
    # Convert to float and normalize if needed
    if normalize:
        flash = img_as_float(flash)
        noflash = img_as_float(noflash)
    
    return flash, noflash

# MARKDOWN CELL: Introduction
# # Flash/No-Flash Photography Project
# 
# This notebook demonstrates the key techniques from the flash/no-flash photography paper.
# We'll explore:
# 
# 1. **Bilateral Filtering** - Remove noise while preserving edges
# 2. **Joint/Cross Bilateral Filtering** - Use flash image to guide the filtering of no-flash image
# 3. **Detail Transfer** - Enhance the no-flash image with details from the flash image
# 4. **Gradient Domain Processing** - Handle complex shadows and specular highlights
#
# Let's begin by loading a sample image pair:

# CODE CELL: Load Sample Images
# Choose an appropriate sample pair, either bilateral or gradient
flash_path = "sample_images/bilateral/books/flash.jpg"
noflash_path = "sample_images/bilateral/books/noflash.jpg"

# Load images
flash_img, noflash_img = load_image_pair(flash_path, noflash_path)

# Display the flash/no-flash pair
display_images(
    [flash_img, noflash_img], 
    titles=["Flash Image", "No-Flash Image"],
    cmaps=['gray', 'gray']
)

# MARKDOWN CELL: Part 1 - Bilateral Filtering
# # Part 1: Bilateral Filtering for Noise Reduction
# 
# Bilateral filtering is a non-linear technique that can reduce noise while preserving edges.
# Let's see how the basic bilateral filter works:

# CODE CELL: Basic Bilateral Filtering
# Create bilateral filter object
bilateral_filter = BilateralFilter()

# Apply standard bilateral filtering to the noisy no-flash image
# This is slow, so we'll use a small spatial sigma
spatial_sigma = 3.0  # Controls the spatial extent of the kernel
intensity_sigma = 0.1  # Controls the range/intensity falloff
filtered_noflash = bilateral_filter.apply(
    noflash_img, 
    spatial_sigma=spatial_sigma, 
    intensity_sigma=intensity_sigma
)

# Display results
display_images(
    [noflash_img, filtered_noflash],
    titles=["Original No-Flash (Noisy)", "Bilateral Filtered"],
    cmaps=['gray', 'gray']
)

# MARKDOWN CELL: Effect of Bilateral Filter Parameters
# ## Effect of Bilateral Filter Parameters
# 
# Let's explore how different parameter values affect the bilateral filter:
# - **Spatial Sigma (σs)**: Controls the size of the spatial neighborhood
# - **Intensity Sigma (σr)**: Controls how much intensity differences are penalized

# CODE CELL: Bilateral Parameter Analysis
# Create a grid of different parameters
spatial_sigmas = [2.0, 5.0, 10.0]
intensity_sigmas = [0.05, 0.1, 0.3]

# Create a figure with 3x3 grid
fig, axes = plt.subplots(len(spatial_sigmas), len(intensity_sigmas), figsize=(15, 15))

# Process with different parameter combinations
for i, ss in enumerate(spatial_sigmas):
    for j, rs in enumerate(intensity_sigmas):
        # Apply bilateral filter with current parameters
        filtered = bilateral_filter.apply(
            noflash_img, 
            spatial_sigma=ss, 
            intensity_sigma=rs
        )
        
        # Display in the appropriate grid cell
        if len(filtered.shape) == 3 and filtered.shape[2] == 3:
            axes[i, j].imshow(np.clip(filtered, 0, 1))
        else:
            axes[i, j].imshow(np.clip(filtered, 0, 1), cmap='gray')
            
        axes[i, j].set_title(f'σs={ss}, σr={rs}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()

# MARKDOWN CELL: Joint Bilateral Filtering
# # Joint Bilateral Filtering
# 
# The key innovation in flash/no-flash photography is using the flash image to guide the filtering
# of the no-flash image. This is called **joint (or cross) bilateral filtering**.
#
# The flash image has good edge information but harsh lighting, while the no-flash image has
# better lighting but more noise. We use the flash image's edge information to preserve details
# while filtering out noise in the no-flash image.

# CODE CELL: Joint Bilateral Filtering
# Apply joint bilateral filtering using flash image as guidance
joint_filtered = bilateral_filter.apply_joint(
    noflash_img,  # Target image to be filtered
    flash_img,    # Edge image for guidance
    spatial_sigma=5.0, 
    intensity_sigma=0.1
)

# Display the results
display_images(
    [flash_img, noflash_img, joint_filtered], 
    titles=["Flash (Edge Guide)", "No-Flash (Noisy)", "Joint Bilateral Result"],
    cmaps=['gray', 'gray', 'gray']
)

# MARKDOWN CELL: Detail Transfer
# # Detail Transfer
# 
# We can enhance the no-flash image by transferring details from the flash image.
# This gives us the best of both: natural lighting from the no-flash image and 
# fine details from the flash image.

# CODE CELL: Detail Transfer
def detail_transfer(flash_img, noflash_img, base_img, alpha=0.5):
    """
    Transfer details from flash to no-flash image
    
    Args:
        flash_img: Flash image (source of details)
        noflash_img: No-flash image (target for transfer)
        base_img: Base image (typically filtered no-flash)
        alpha: Detail transfer strength
        
    Returns:
        Enhanced image with details
    """
    # Convert to luminance if color images
    if len(flash_img.shape) == 3 and flash_img.shape[2] == 3:
        flash_lum = color.rgb2gray(flash_img)
        base_lum = color.rgb2gray(base_img)
    else:
        flash_lum = flash_img
        base_lum = base_img
    
    # Extract details from flash image (detail layer)
    flash_details = flash_lum / (base_lum + 1e-6)  # Detail quotient
    
    # Apply details to the filtered no-flash image
    if len(noflash_img.shape) == 3 and noflash_img.shape[2] == 3:
        # For color images, apply to each channel
        enhanced = np.zeros_like(noflash_img)
        for i in range(3):
            enhanced[:,:,i] = joint_filtered[:,:,i] * (flash_details ** alpha)
    else:
        # For grayscale images
        enhanced = joint_filtered * (flash_details ** alpha)
    
    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced

# Apply detail transfer
enhanced_img = detail_transfer(
    flash_img, 
    noflash_img, 
    joint_filtered,
    alpha=0.5  # Adjust detail strength
)

# Display results
display_images(
    [noflash_img, joint_filtered, enhanced_img], 
    titles=["Original No-Flash", "Joint Bilateral", "With Detail Transfer"],
    cmaps=['gray', 'gray', 'gray']
)

# MARKDOWN CELL: Part 2 - Gradient Domain Processing
# # Part 2: Gradient Domain Processing
# 
# For scenes with significant shadows or specular highlights, we need a more advanced approach.
# Gradient domain processing involves:
# 
# 1. Computing gradients of both images
# 2. Creating a fused gradient field
# 3. Solving a Poisson equation to reintegrate the image
#
# Let's load a more challenging image pair:

# CODE CELL: Load Gradient Sample Images
# Use a sample pair with challenging shadows/highlights
flash_path_grad = "sample_images/gradient/plant/flash.jpg"
noflash_path_grad = "sample_images/gradient/plant/noflash.jpg"

# Load images
flash_img_grad, noflash_img_grad = load_image_pair(flash_path_grad, noflash_path_grad)

# Display the flash/no-flash pair
display_images(
    [flash_img_grad, noflash_img_grad], 
    titles=["Flash Image (with shadows)", "No-Flash Image"],
    cmaps=['gray', 'gray']
)

# MARKDOWN CELL: Poisson Solver
# ## Poisson Solver Overview
# 
# The gradient domain approach solves the Poisson equation to reintegrate an image from a
# manipulated gradient field. This is powerful for handling shadows and highlights.
#
# Let's see a simple test by differentiating and reintegrating an image:

# CODE CELL: Differentiate and Reintegrate Test
# Create Poisson solver
poisson_solver = PoissonSolver()

# Simple test: differentiate and reintegrate
# This should give us back roughly the same image
reintegrated = poisson_solver.differentiate_and_reintegrate(
    noflash_img_grad,
    epsilon=1e-6,
    max_iterations=1000
)

# Display results
display_images(
    [noflash_img_grad, reintegrated], 
    titles=["Original Image", "Differentiated & Reintegrated"],
    cmaps=['gray', 'gray']
)

# MARKDOWN CELL: Gradient Fusion Process
# ## Gradient Fusion for Flash/No-Flash
# 
# Now let's apply the gradient fusion algorithm to combine the best aspects of both images:

# CODE CELL: Gradient Fusion
# Generate fused gradient field
# This combines the gradients from flash and no-flash images
# with suitable weights based on coherency and saturation
fused_gradients = poisson_solver.generate_fused_gradient(
    noflash_img_grad,  # Ambient image 
    flash_img_grad,    # Flash image
    sigma=5.0,         # Parameter for saturation weight
    tau_s=0.12         # Threshold for saturation weight
)

# MARKDOWN CELL: Coherency and Saturation Maps
# ## Coherency and Saturation Maps
# 
# The algorithm computes two important maps:
# 1. **Coherency Map (M)**: Measures how aligned the gradients are between flash and no-flash images
# 2. **Saturation Weight (w_s)**: Identifies over/under saturated regions in the flash image
#
# Let's visualize these:

# CODE CELL: Visualize Gradient Maps
# Extract the coherency map and saturation weight
coherency_map = fused_gradients["coherency_map"]
saturation_weight = fused_gradients["saturation_weight"]

# Visualize maps
display_images(
    [coherency_map, saturation_weight], 
    titles=["Coherency Map", "Saturation Weight Map"],
    cmaps=['viridis', 'plasma']
)

# MARKDOWN CELL: Gradient Integration
# ## Final Integration
# 
# Now we reintegrate the fused gradient field to get our final result:

# CODE CELL: Integrate Fused Gradient
# Integrate the fused gradient to produce the final image
final_image = poisson_solver.integrate_fused_gradient(
    fused_gradients["fused_grad_x"],  # X component of fused gradient
    fused_gradients["fused_grad_y"],  # Y component of fused gradient
    noflash_img_grad,                 # Ambient image (for boundary conditions)
    flash_img_grad,                   # Flash image
    boundary_type="ambient",          # Use ambient image for boundaries
    init_type="ambient",              # Start from ambient image 
    epsilon=1e-6,                     # Convergence parameter
    max_iterations=1000               # Maximum iterations
)

# Display all results
display_images(
    [flash_img_grad, noflash_img_grad, final_image], 
    titles=["Flash Image", "No-Flash Image", "Gradient Domain Result"],
    cmaps=['gray', 'gray', 'gray']
)

# MARKDOWN CELL: Comparison of Approaches
# # Comparison Between Bilateral and Gradient Domain Approaches
# 
# Let's compare both approaches we've explored:
# 
# | Aspect | Bilateral Filtering | Gradient Domain |
# |--------|---------------------|-----------------|
# | Strengths | Fast, good for noise reduction | Handles shadows and highlights well |
# | Weaknesses | Struggles with shadows/highlights | Computationally intensive |
# | Best for | Indoor scenes, portraits, simple lighting | Complex lighting, specular highlights |
#
# Let's see visual comparisons:

# CODE CELL: Visual Comparison
# Load a bilateral example pair
flash_bilateral, noflash_bilateral = load_image_pair(flash_path, noflash_path)

# Process with joint bilateral filtering
bilateral_result = bilateral_filter.apply_joint(
    noflash_bilateral,
    flash_bilateral,
    spatial_sigma=5.0, 
    intensity_sigma=0.1
)

# Add detail transfer
bilateral_result_with_detail = detail_transfer(
    flash_bilateral, 
    noflash_bilateral, 
    bilateral_result,
    alpha=0.5
)

# Display results for bilateral filtering approach
display_images(
    [flash_bilateral, noflash_bilateral, bilateral_result_with_detail], 
    titles=["Flash", "No-Flash", "Bilateral Result"],
    cmaps=['gray', 'gray', 'gray']
)

# Display results for gradient domain approach
display_images(
    [flash_img_grad, noflash_img_grad, final_image], 
    titles=["Flash", "No-Flash", "Gradient Domain Result"],
    cmaps=['gray', 'gray', 'gray']
)

# MARKDOWN CELL: Custom Examples
# # Try Your Own Examples
# 
# You can try these techniques on your own flash/no-flash image pairs.
# Here's how to load and process your own images:

# CODE CELL: Custom Example
# Replace these with your own image paths
# my_flash_path = "path/to/your/flash/image.jpg" 
# my_noflash_path = "path/to/your/noflash/image.jpg"

# Uncomment to load and process your images
# my_flash, my_noflash = load_image_pair(my_flash_path, my_noflash_path)

# Option 1: Bilateral Filtering
# my_bilateral_result = bilateral_filter.apply_joint(
#     my_noflash, my_flash, spatial_sigma=5.0, intensity_sigma=0.1
# )
# my_bilateral_with_detail = detail_transfer(
#     my_flash, my_noflash, my_bilateral_result, alpha=0.5
# )

# Option 2: Gradient Domain Processing
# my_fused_gradients = poisson_solver.generate_fused_gradient(
#     my_noflash, my_flash, sigma=5.0, tau_s=0.12
# )
# my_gradient_result = poisson_solver.integrate_fused_gradient(
#     my_fused_gradients["fused_grad_x"], 
#     my_fused_gradients["fused_grad_y"],
#     my_noflash, my_flash, 
#     boundary_type="ambient", init_type="ambient"
# )

# display_images(
#     [my_flash, my_noflash, my_gradient_result],
#     titles=["Flash", "No-Flash", "Result"]
# )

# MARKDOWN CELL: Conclusion
# # Conclusion
# 
# In this project, we've explored two main approaches to flash/no-flash photography:
# 
# 1. **Bilateral Filtering**: Effective for noise reduction while preserving edges
#    - Joint bilateral filtering uses flash image to guide filtering
#    - Detail transfer enhances the result with fine details
#
# 2. **Gradient Domain Processing**: Handles complex lighting situations
#    - Fuses gradients from both images based on coherency and saturation
#    - Solves Poisson equation to reintegrate the final image
#
# Each approach has its strengths and is suited for different scenarios.
# The flash/no-flash photography technique provides a powerful way to capture
# high-quality images in low-light environments without the harsh lighting
# typically associated with flash photography.

if __name__ == "__main__":
    print("Running Flash/No-Flash Photography Demo...")
    # This script is designed to be run in a Jupyter notebook
    # If run as a standalone script, it will execute all cells sequentially 