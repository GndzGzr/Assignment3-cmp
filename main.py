#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash and No-Flash Photography Processing Demo

This script implements techniques from the papers:
1. Petschnigg et al. "Digital Photography with Flash and No-Flash Image Pairs"
2. Eisemann and Durand "Flash Photography Enhancement via Intrinsic Relighting"
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from classes_functions import ImageUtils, BilateralFilter, GradientDomainProcessor

def main():
    # Initialize utility classes
    image_utils = ImageUtils()
    bilateral_filter = BilateralFilter()
    gradient_processor = GradientDomainProcessor()

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Create the images directory if it doesn't exist
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)

    # Load a flash/no-flash pair
    flash_filename = "flash.jpg"  # Replace with your flash image filename
    noflash_filename = "noflash.jpg"  # Replace with your no-flash image filename

    print("Looking for images in:", os.path.abspath(images_dir))
    
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
        plt.savefig(os.path.join(output_dir, "input_images.png"))
        plt.close()
        
        print("Processing with bilateral filtering...")
        # Process with bilateral filtering
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
        
        print("Processing with gradient-domain technique...")
        # Process with gradient-domain technique
        gradient_results = gradient_processor.process_flash_no_flash_pair(
            ambient, flash,
            sigma=5.0,              # Parameter for weight calculation
            tau_s=0.12,             # Threshold for saturation weight calculation
            boundary_type="ambient", # Type of boundary conditions
            init_type="average",    # Type of initialization for the solver
            epsilon=1e-6,           # Convergence parameter
            max_iterations=1000     # Maximum number of iterations
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
        
        print("Completed processing! Results saved to:", os.path.abspath(output_dir))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please place your flash and no-flash images in the '{images_dir}' folder.")
        print(f"Required files: {flash_filename} and {noflash_filename}")
        
        # Create a README file with instructions
        with open(os.path.join(images_dir, "README.txt"), "w") as f:
            f.write("Flash/No-Flash Photography Processing\n")
            f.write("====================================\n\n")
            f.write("Please place your flash and no-flash image pairs in this folder.\n")
            f.write(f"Rename them to '{flash_filename}' and '{noflash_filename}' respectively.\n\n")
            f.write("Guidelines for capturing effective flash/no-flash pairs:\n")
            f.write("1. Use a stable tripod to ensure both images are perfectly aligned\n")
            f.write("2. Use manual focus and exposure settings\n")
            f.write("3. Set a fixed white balance (not auto)\n")
            f.write("4. The flash image should be properly exposed (not overexposed)\n")
            f.write("5. For the no-flash image, use a longer exposure time\n")
            f.write("6. Avoid highly reflective or transparent objects (they cause flash artifacts)\n")
        
        print(f"Created README.txt in the {images_dir} folder with instructions.")

if __name__ == "__main__":
    main() 