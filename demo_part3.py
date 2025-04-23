#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash/No-Flash Photography - Part 3 Demo

This script demonstrates processing of flash/no-flash image pairs using both:
1. Bilateral filtering (for denoising)
2. Gradient domain processing (for fusion)

Usage:
    python demo_part3.py

First run download_sample_pairs.py to get sample images.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from classes_functions import ImageUtils, BilateralFilter, GradientDomainProcessor

def process_bilateral_pair(data_dir, flash_name, noflash_name, output_dir, 
                          sigma_s=16.0, sigma_r=0.1, epsilon=0.02):
    """Process a flash/no-flash pair using bilateral filtering."""
    print(f"\nProcessing bilateral filtering example from {data_dir}")
    
    # Create output directory
    pair_output_dir = os.path.join(output_dir, os.path.basename(data_dir) + "_bilateral")
    os.makedirs(pair_output_dir, exist_ok=True)
    
    # Initialize utility classes
    image_utils = ImageUtils()
    bilateral_filter = BilateralFilter()
    
    # Load images
    try:
        ambient, flash = image_utils.load_flash_no_flash_pair(data_dir, flash_name, noflash_name)
        print("Images loaded successfully")
    except Exception as e:
        print(f"Error loading images: {e}")
        return None
    
    # Process with bilateral filtering
    print("Applying bilateral filtering...")
    bilateral_results = bilateral_filter.process_flash_no_flash_pair(
        ambient, flash,
        sigma_s_basic=sigma_s,      # Spatial sigma for basic bilateral filter
        sigma_r_basic=sigma_r,      # Range sigma for basic bilateral filter
        sigma_s_joint=sigma_s,      # Spatial sigma for joint bilateral filter
        sigma_r_joint=sigma_r,      # Range sigma for joint bilateral filter
        epsilon=epsilon,            # Detail layer strength
        shadow_thresh=0.1,          # Threshold for shadow detection
        spec_thresh=0.9             # Threshold for specularity detection
    )
    
    # Save the results
    result_path = os.path.join(pair_output_dir, "bilateral_results.png")
    image_utils.plot_comparison(bilateral_results, save_dir=pair_output_dir)
    image_utils.save_image(bilateral_results["a_final"], 
                          os.path.join(pair_output_dir, "bilateral_final.jpg"))
    
    # Create analysis figure with parameter variations
    analyze_bilateral_parameters(ambient, flash, pair_output_dir, 
                               sigma_s_values=[8.0, 16.0, 32.0], 
                               sigma_r_values=[0.05, 0.1, 0.2],
                               epsilon_values=[0.01, 0.02, 0.04])
    
    print(f"Bilateral filtering results saved to {pair_output_dir}")
    return bilateral_results

def process_gradient_pair(data_dir, flash_name, noflash_name, output_dir,
                         sigma=5.0, tau_s=0.12):
    """Process a flash/no-flash pair using gradient domain processing."""
    print(f"\nProcessing gradient domain example from {data_dir}")
    
    # Create output directory
    pair_output_dir = os.path.join(output_dir, os.path.basename(data_dir) + "_gradient")
    os.makedirs(pair_output_dir, exist_ok=True)
    
    # Initialize utility classes
    image_utils = ImageUtils()
    gradient_processor = GradientDomainProcessor()
    
    # Load images
    try:
        ambient, flash = image_utils.load_flash_no_flash_pair(data_dir, flash_name, noflash_name)
        print("Images loaded successfully")
    except Exception as e:
        print(f"Error loading images: {e}")
        return None
    
    # Process with gradient-domain technique
    print("Applying gradient domain processing...")
    gradient_results = gradient_processor.process_flash_no_flash_pair(
        ambient, flash,
        sigma=sigma,                 # Parameter for weight calculation
        tau_s=tau_s,                 # Threshold for saturation weight calculation
        boundary_type="ambient",     # Type of boundary conditions
        init_type="average",         # Type of initialization for the solver
        epsilon=1e-6,                # Convergence parameter
        max_iterations=1000          # Maximum number of iterations
    )
    
    # Save the results
    gradient_processor.plot_results(gradient_results, 
                                  save_path=os.path.join(pair_output_dir, "gradient_results.png"))
    
    gradient_processor.plot_gradient_fields(
        ambient, flash, gradient_results["gradient_fields"],
        save_path=os.path.join(pair_output_dir, "gradient_fields.png")
    )
    
    image_utils.save_image(gradient_results["fused_image"], 
                          os.path.join(pair_output_dir, "gradient_final.jpg"))
    
    # Create analysis figure with parameter variations
    analyze_gradient_parameters(ambient, flash, pair_output_dir,
                               sigma_values=[1.0, 5.0, 10.0],
                               tau_s_values=[0.05, 0.12, 0.2])
    
    print(f"Gradient domain results saved to {pair_output_dir}")
    return gradient_results

def analyze_bilateral_parameters(ambient, flash, output_dir, 
                               sigma_s_values, sigma_r_values, epsilon_values):
    """Create analysis figures for bilateral filtering with different parameters."""
    print("Creating bilateral parameter analysis figures...")
    
    # Create a subdirectory for parameter analysis
    analysis_dir = os.path.join(output_dir, "parameter_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    bilateral_filter = BilateralFilter()
    image_utils = ImageUtils()
    
    # 1. Analyze spatial sigma (sigma_s) effect
    plt.figure(figsize=(15, 10))
    
    # Original ambient and flash images
    plt.subplot(2, 3, 1)
    plt.imshow(ambient)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(flash)
    plt.title("Flash")
    plt.axis('off')
    
    # Different sigma_s values
    for i, sigma_s in enumerate(sigma_s_values):
        results = bilateral_filter.process_flash_no_flash_pair(
            ambient, flash,
            sigma_s_basic=sigma_s, sigma_r_basic=0.1,
            sigma_s_joint=sigma_s, sigma_r_joint=0.1,
            epsilon=0.02
        )
        
        plt.subplot(2, 3, i+3)
        plt.imshow(results["a_final"])
        plt.title(f"Spatial Sigma = {sigma_s}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "bilateral_sigma_s_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Analyze range sigma (sigma_r) effect
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(ambient)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(flash)
    plt.title("Flash")
    plt.axis('off')
    
    for i, sigma_r in enumerate(sigma_r_values):
        results = bilateral_filter.process_flash_no_flash_pair(
            ambient, flash,
            sigma_s_basic=16.0, sigma_r_basic=sigma_r,
            sigma_s_joint=16.0, sigma_r_joint=sigma_r,
            epsilon=0.02
        )
        
        plt.subplot(2, 3, i+3)
        plt.imshow(results["a_final"])
        plt.title(f"Range Sigma = {sigma_r}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "bilateral_sigma_r_comparison.png"), dpi=150)
    plt.close()
    
    # 3. Analyze detail strength (epsilon) effect
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(ambient)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(flash)
    plt.title("Flash")
    plt.axis('off')
    
    for i, epsilon in enumerate(epsilon_values):
        results = bilateral_filter.process_flash_no_flash_pair(
            ambient, flash,
            sigma_s_basic=16.0, sigma_r_basic=0.1,
            sigma_s_joint=16.0, sigma_r_joint=0.1,
            epsilon=epsilon
        )
        
        plt.subplot(2, 3, i+3)
        plt.imshow(results["a_final"])
        plt.title(f"Detail Strength = {epsilon}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "bilateral_epsilon_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Bilateral parameter analysis saved to {analysis_dir}")

def analyze_gradient_parameters(ambient, flash, output_dir, 
                               sigma_values, tau_s_values):
    """Create analysis figures for gradient domain processing with different parameters."""
    print("Creating gradient domain parameter analysis figures...")
    
    # Create a subdirectory for parameter analysis
    analysis_dir = os.path.join(output_dir, "parameter_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    gradient_processor = GradientDomainProcessor()
    image_utils = ImageUtils()
    
    # 1. Analyze sigma effect
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(ambient)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(flash)
    plt.title("Flash")
    plt.axis('off')
    
    for i, sigma in enumerate(sigma_values):
        results = gradient_processor.process_flash_no_flash_pair(
            ambient, flash,
            sigma=sigma,
            tau_s=0.12,
            boundary_type="ambient",
            init_type="average"
        )
        
        plt.subplot(2, 3, i+3)
        plt.imshow(results["fused_image"])
        plt.title(f"Sigma = {sigma}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "gradient_sigma_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Analyze tau_s effect
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(ambient)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(flash)
    plt.title("Flash")
    plt.axis('off')
    
    for i, tau_s in enumerate(tau_s_values):
        results = gradient_processor.process_flash_no_flash_pair(
            ambient, flash,
            sigma=5.0,
            tau_s=tau_s,
            boundary_type="ambient",
            init_type="average"
        )
        
        plt.subplot(2, 3, i+3)
        plt.imshow(results["fused_image"])
        plt.title(f"Tau_s = {tau_s}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "gradient_tau_s_comparison.png"), dpi=150)
    plt.close()
    
    # 3. Analyze boundary conditions effect
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(ambient)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(flash)
    plt.title("Flash")
    plt.axis('off')
    
    boundary_types = ["ambient", "flash", "average"]
    for i, boundary_type in enumerate(boundary_types):
        results = gradient_processor.process_flash_no_flash_pair(
            ambient, flash,
            sigma=5.0,
            tau_s=0.12,
            boundary_type=boundary_type,
            init_type="average"
        )
        
        plt.subplot(2, 3, i+3)
        plt.imshow(results["fused_image"])
        plt.title(f"Boundary: {boundary_type}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "gradient_boundary_comparison.png"), dpi=150)
    plt.close()
    
    print(f"Gradient parameter analysis saved to {analysis_dir}")

def create_comparison_report(output_dir):
    """Create a report comparing both techniques."""
    report_path = os.path.join(output_dir, "technique_comparison.md")
    
    with open(report_path, "w") as f:
        f.write("# Flash/No-Flash Photography Technique Comparison\n\n")
        
        f.write("## Bilateral Filtering vs. Gradient-Domain Processing\n\n")
        
        f.write("### Bilateral Filtering\n")
        f.write("- **Advantages**:\n")
        f.write("  - Faster computation time\n")
        f.write("  - Effective noise reduction while preserving edges\n")
        f.write("  - Simpler implementation with fewer parameters\n")
        f.write("  - Good for detail enhancement from flash image\n\n")
        
        f.write("- **Disadvantages**:\n")
        f.write("  - Can produce halo artifacts around strong edges\n")
        f.write("  - Less effective for shadow and specular handling\n")
        f.write("  - May require more parameter tuning for optimal results\n\n")
        
        f.write("- **Best Use Cases**:\n")
        f.write("  - Denoising low-light images\n")
        f.write("  - Indoor photography in dimly lit environments\n")
        f.write("  - When computational resources are limited\n\n")
        
        f.write("### Gradient-Domain Processing\n")
        f.write("- **Advantages**:\n")
        f.write("  - Better preservation of edge transitions\n")
        f.write("  - More natural handling of shadows and highlights\n")
        f.write("  - Less prone to halo artifacts\n")
        f.write("  - Better for complex scenes with mixed lighting\n\n")
        
        f.write("- **Disadvantages**:\n")
        f.write("  - More computationally intensive\n")
        f.write("  - Requires solving Poisson equation (iterative process)\n")
        f.write("  - More complex implementation\n")
        f.write("  - Results depend on proper boundary conditions\n\n")
        
        f.write("- **Best Use Cases**:\n")
        f.write("  - Scenes with strong specular highlights\n")
        f.write("  - Complex lighting with shadows cast by flash\n")
        f.write("  - When highest quality results are needed\n\n")
        
        f.write("## Parameter Analysis\n\n")
        
        f.write("### Bilateral Filtering Parameters\n")
        f.write("- **Spatial Sigma (sigma_s)**:\n")
        f.write("  - Controls the spatial extent of the filter\n")
        f.write("  - Larger values blur over larger regions\n")
        f.write("  - Typical values: 8-32 pixels\n\n")
        
        f.write("- **Range Sigma (sigma_r)**:\n")
        f.write("  - Controls edge preservation\n")
        f.write("  - Smaller values preserve stronger edges\n")
        f.write("  - Typical values: 0.05-0.2 (for [0,1] range images)\n\n")
        
        f.write("- **Detail Strength (epsilon)**:\n")
        f.write("  - Controls amount of detail transferred from flash image\n")
        f.write("  - Higher values increase detail but may introduce noise\n")
        f.write("  - Typical values: 0.01-0.05\n\n")
        
        f.write("### Gradient-Domain Processing Parameters\n")
        f.write("- **Sigma**:\n")
        f.write("  - Controls weight calculation for gradient fusion\n")
        f.write("  - Higher values increase flash influence on gradients\n")
        f.write("  - Typical values: 1-10\n\n")
        
        f.write("- **Tau_s (Saturation threshold)**:\n")
        f.write("  - Threshold for saturation weight calculation\n")
        f.write("  - Controls how the algorithm handles bright areas\n")
        f.write("  - Typical values: 0.05-0.2\n\n")
        
        f.write("- **Boundary Conditions**:\n")
        f.write("  - Define values at image boundary for Poisson solving\n")
        f.write("  - Options: ambient (no-flash), flash, or average\n")
        f.write("  - Affect color tone of final result\n\n")
        
        f.write("## Guidelines for Capturing Good Flash/No-Flash Pairs\n\n")
        
        f.write("1. **Camera Setup**:\n")
        f.write("   - Use a stable tripod to ensure both images are perfectly aligned\n")
        f.write("   - Use manual focus to keep focus consistent between shots\n")
        f.write("   - Use manual exposure settings for the no-flash (ambient) image\n")
        f.write("   - Set a fixed white balance (not auto)\n\n")
        
        f.write("2. **Flash Control**:\n")
        f.write("   - Use an external flash if possible (more control over flash power)\n")
        f.write("   - The flash image should be properly exposed (not overexposed)\n")
        f.write("   - For the no-flash image, use a longer exposure time\n\n")
        
        f.write("3. **Subject Selection**:\n")
        f.write("   - For bilateral filtering: dimly lit environments with details\n")
        f.write("   - For gradient domain: scenes with mixed specular and matte surfaces\n")
        f.write("   - Avoid scenes with moving objects\n\n")
        
        f.write("4. **Common Issues to Avoid**:\n")
        f.write("   - Camera movement between shots\n")
        f.write("   - Subject movement between shots\n")
        f.write("   - Flash shadows or harsh specular highlights\n")
        f.write("   - Flash image overexposure\n")
        f.write("   - No-flash image too dark or noisy\n")
    
    print(f"Technique comparison report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Flash/No-Flash Photography - Part 3 Demo")
    parser.add_argument("--sample_dir", type=str, default="sample_images",
                       help="Directory containing sample images")
    parser.add_argument("--output_dir", type=str, default="part3_results",
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if sample directory exists
    if not os.path.exists(args.sample_dir):
        print(f"Sample directory '{args.sample_dir}' not found.")
        print("Please run download_sample_pairs.py first to download sample images.")
        print("Example: python download_sample_pairs.py")
        return
    
    print("Flash/No-Flash Photography - Part 3 Demo")
    print("=======================================")
    
    # Process bilateral filtering examples
    bilateral_dir = os.path.join(args.sample_dir, "bilateral")
    if os.path.exists(bilateral_dir):
        for sample in os.listdir(bilateral_dir):
            sample_dir = os.path.join(bilateral_dir, sample)
            if os.path.isdir(sample_dir):
                process_bilateral_pair(sample_dir, "flash.jpg", "noflash.jpg", args.output_dir)
    else:
        print(f"Bilateral examples directory '{bilateral_dir}' not found.")
    
    # Process gradient domain examples
    gradient_dir = os.path.join(args.sample_dir, "gradient")
    if os.path.exists(gradient_dir):
        for sample in os.listdir(gradient_dir):
            sample_dir = os.path.join(gradient_dir, sample)
            if os.path.isdir(sample_dir):
                process_gradient_pair(sample_dir, "flash.jpg", "noflash.jpg", args.output_dir)
    else:
        print(f"Gradient examples directory '{gradient_dir}' not found.")
    
    # Create technique comparison report
    create_comparison_report(args.output_dir)
    
    print("\nProcessing complete!")
    print(f"All results saved to {os.path.abspath(args.output_dir)}")
    print("\nThe key findings and technique comparison are available in:")
    print(f"{os.path.join(os.path.abspath(args.output_dir), 'technique_comparison.md')}")

if __name__ == "__main__":
    main() 