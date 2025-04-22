# Flash/No-Flash Photography - Bilateral Filtering Implementation

This repository contains an implementation of bilateral filtering techniques for flash/no-flash photography as described in the paper by Petschnigg et al. [1]. The implementation focuses on denoising the ambient image using bilateral filtering and enhancing it with details from the flash image.

## Overview

Flash photography is a powerful technique that allows capturing photos in low-light conditions but often produces harsh lighting that ruins the scene's ambiance. Ambient lighting preserves the scene's mood but can result in noisy images. This implementation combines the best of both worlds by:

1. Using basic bilateral filtering to denoise the ambient image
2. Applying joint bilateral filtering guided by the flash image
3. Transferring details from the flash image to the ambient image
4. Creating a mask to handle shadows and specularities in the flash image

## Project Structure

- `classes_functions/` - Core implementations
  - `bilateral_filter.py` - Bilateral filtering algorithms
  - `image_utils.py` - Utilities for image loading, saving, and visualization
- `demo_bilateral_filter.py` - Main demo script
- `batch_processing.py` - Script for batch processing multiple images
- `parameter_exploration.py` - Script for exploring different parameter values

## Requirements

The implementation requires the following dependencies:
- Python 3.6+
- NumPy
- OpenCV (with contrib modules for joint bilateral filtering)
- Matplotlib
- Argparse

You can install the required packages using:

```bash
pip install numpy opencv-python opencv-contrib-python matplotlib argparse
```

## Dataset

The implementation expects a dataset with the following structure:
```
data/
  camera/
    flash/
      image1.jpg
      image2.jpg
      ...
    nonflash/
      image1.jpg
      image2.jpg
      ...
```

The provided code is designed to work with the flash/no-flash image pairs from the paper by Petschnigg et al. [1].

## Usage

### Basic Demo

To run the basic demo on a single image pair:

```bash
python demo_bilateral_filter.py --data_dir data/camera --image_name cave-flash.jpg
```

This will process the specified flash/no-flash image pair and save the results in the `results` directory.

### Parameter Exploration

To explore the effect of different parameter values:

```bash
python parameter_exploration.py --data_dir data/camera --image_name cave-flash.jpg
```

This will generate results for various parameter values and save them in the `results/param_exploration` directory.

### Batch Processing

To process all image pairs in the dataset:

```bash
python batch_processing.py --data_dir data/camera
```

This will process all flash/no-flash image pairs in the dataset and save the results in the `results/batch` directory.

## Parameters

The implementation allows customizing the following parameters:

- `sigma_s_basic` - Spatial sigma for basic bilateral filtering (default: 8.0)
- `sigma_r_basic` - Range sigma for basic bilateral filtering (default: 0.1)
- `sigma_s_joint` - Spatial sigma for joint bilateral filtering (default: 8.0)
- `sigma_r_joint` - Range sigma for joint bilateral filtering (default: 0.1)
- `epsilon` - Small constant for detail transfer (default: 0.02)
- `shadow_thresh` - Threshold for shadow detection (default: 0.1)
- `spec_thresh` - Threshold for specularity detection (default: 0.9)

## Results

The implementation produces the following outputs:

1. `a_base`: Result of basic bilateral filtering on the ambient image
2. `a_nr`: Result of joint bilateral filtering
3. `a_detail`: Result after detail transfer
4. `a_final`: Final result after shadow and specularity masking
5. Various difference images to visualize the effect of each step

## References

[1] Petschnigg, G., Szeliski, R., Agrawala, M., Cohen, M., Hoppe, H., & Toyama, K. (2004). Digital photography with flash and no-flash image pairs. ACM Transactions on Graphics (TOG), 23(3), 664-672.

[2] Durand, F., & Dorsey, J. (2002). Fast bilateral filtering for the display of high-dynamic-range images. ACM Transactions on Graphics (TOG), 21(3), 257-266. 