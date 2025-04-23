#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Flash and No-Flash Photography Processing

This notebook implements techniques from the papers:
1. Petschnigg et al. "Digital Photography with Flash and No-Flash Image Pairs"
2. Eisemann and Durand "Flash Photography Enhancement via Intrinsic Relighting"
3. Agrawal et al. "Removing Photography Artifacts using Gradient Projection and Flash-Exposure Sampling"

## Setup
First, import the necessary libraries and helper functions.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from classes_functions import ImageUtils, BilateralFilter, GradientDomainProcessor

# Set up matplotlib for inline plotting
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['image.cmap'] = 'viridis'

# Initialize utility classes
image_utils = ImageUtils()
bilateral_filter = BilateralFilter()
gradient_processor = GradientDomainProcessor()

"""
## Capturing Effective Flash/No-Flash Pairs

### Guidelines for Capturing:
1. **Camera Setup**:
   - Use a stable tripod to ensure both images are perfectly aligned
   - Use manual focus to keep focus consistent between shots
   - Use manual exposure settings for the no-flash (ambient) image
   - Set a fixed white balance (not auto)

2. **Subject and Environment**:
   - Choose a subject with interesting texture details
   - Scene should be dimly lit but not completely dark
   - Avoid highly reflective or transparent objects (they cause flash artifacts)
   - Keep a reasonable distance from the subject

3. **Flash Control**:
   - Use an external flash if possible (more control over flash power)
   - The flash image should be properly exposed (not overexposed)
   - For the no-flash image, use a longer exposure time to compensate for lack of flash

4. **Common Issues to Avoid**:
   - Camera movement between shots (use a remote trigger if possible)
   - Subject movement between shots (ask people to stay still)
   - Flash shadows or specular highlights (adjust flash angle or use a diffuser)
   - Flash image overexposure (reduce flash power or increase distance)
   - No-flash image too dark or noisy (increase exposure time or ISO)

### Bilateral Filtering vs. Gradient-Domain Processing:
- **Bilateral Filtering**:
  - Faster and easier to implement
  - Good for extracting details from the flash image
  - Handles shadows and highlights well when properly tuned
  - May produce halos or artifacts around strong edges

- **Gradient-Domain Processing**:
  - Better preservation of edge transitions
  - Handles shadows and highlights more naturally
  - More computationally intensive
  - Results depend heavily on proper Poisson equation solving
"""

# Define a directory for saving output images
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

"""
## Load an Example Flash/No-Flash Pair

You should download a pair of flash and no-flash images or use your own captured images.
Place them in a folder named "images" with appropriate filenames.
"""

# Load a flash/no-flash pair (adjust the filenames as needed)
images_dir = "images"
flash_filename = "flash.jpg"  # Replace with your flash image filename
noflash_filename = "noflash.jpg"  # Replace with your no-flash image filename

# Create the images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

"""
When you have your images ready, uncomment the following code to load them:

```python
try:
    ambient, flash = image_utils.load_flash_no_flash_pair(images_dir, flash_filename, noflash_filename)
    
    # Display the loaded images
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(ambient)
    plt.title('Ambient (No Flash)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(flash)
    plt.title('Flash')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please place your flash and no-flash images in the 'images' folder.")
```

## Process with Bilateral Filtering

First, we'll apply the bilateral filtering technique to enhance the no-flash image using details from the flash image.
"""

"""
When your images are loaded, uncomment this code to process them using bilateral filtering:

```python
# Process the images with bilateral filtering
bilateral_results = bilateral_filter.process_flash_no_flash_pair(
    ambient, flash,
    sigma_s_basic=16.0,  # Spatial sigma for basic bilateral filter
    sigma_r_basic=0.1,   # Range sigma for basic bilateral filter
    sigma_s_joint=16.0,  # Spatial sigma for joint bilateral filter
    sigma_r_joint=0.1,   # Range sigma for joint bilateral filter
    epsilon=0.02,        # Detail layer strength
    shadow_thresh=0.1,   # Threshold for shadow detection
    spec_thresh=0.9      # Threshold for specularity detection
)

# Visualize the results
image_utils.plot_comparison(bilateral_results, save_dir=output_dir)

# Save the final result
image_utils.save_image(bilateral_results["a_final"], os.path.join(output_dir, "bilateral_result.jpg"))
```

## Process with Gradient-Domain Technique

Now we'll apply the gradient-domain technique to fuse the images in gradient space.
"""

"""
When your images are loaded, uncomment this code to process them using gradient-domain fusion:

```python
# Process the images with gradient-domain processing
gradient_results = gradient_processor.process_flash_no_flash_pair(
    ambient, flash,
    sigma=5.0,             # Parameter for weight calculation
    tau_s=0.12,            # Threshold for saturation weight calculation
    boundary_type="ambient", # Type of boundary conditions
    init_type="average",   # Type of initialization for the solver
    epsilon=1e-6,          # Convergence parameter
    max_iterations=1000    # Maximum number of iterations
)

# Visualize the results
gradient_processor.plot_results(gradient_results, save_path=os.path.join(output_dir, "gradient_results.png"))

# Visualize the gradient fields for better understanding
gradient_processor.plot_gradient_fields(
    ambient, flash, gradient_results["gradient_fields"],
    save_path=os.path.join(output_dir, "gradient_fields.png")
)

# Save the final result
image_utils.save_image(gradient_results["fused_image"], os.path.join(output_dir, "gradient_result.jpg"))
```

## Analysis

Analyze the results from both techniques and compare them.
- How do the bilateral filtering and gradient-domain results differ?
- Which technique better preserves details from the flash image?
- How well are shadows and specularities handled in each technique?
- What artifacts or issues do you notice in each result?

## Experiment with Parameters

Try adjusting the parameters to see how they affect the results:
- Increase/decrease the spatial sigma (`sigma_s`) to affect the blur radius
- Increase/decrease the range sigma (`sigma_r`) to affect the edge-preserving quality
- Adjust the detail layer strength (`epsilon`) to control detail transfer
- Modify shadow and specularity thresholds to improve detection

## Conclusion

Flash/no-flash photography processing techniques help in combining the best aspects of both images:
- The natural lighting and colors from the ambient (no-flash) image
- The low noise and sharp details from the flash image

These techniques are particularly useful for:
- Low-light photography without the harsh appearance of direct flash
- Indoor photography where ambient light produces pleasing colors but high noise
- Situations where a tripod is available for capturing aligned image pairs

## Tips for Better Results:
1. Ensure perfect alignment between flash and no-flash images
2. Experiment with different parameter settings for your specific images
3. For scenes with significant shadows, the gradient-domain technique often performs better
4. For general enhancement, bilateral filtering is faster and often sufficient
""" 