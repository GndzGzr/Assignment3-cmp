"""
Flash/No-Flash Photography Demo
===============================

This script demonstrates the implementation of bilateral filtering techniques for flash/no-flash
photography as described in the paper by Petschnigg et al.

It demonstrates:
1. Basic bilateral filtering on the ambient image
2. Joint bilateral filtering using flash image
3. Detail transfer from flash to ambient image
4. Shadow and specularity masking
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from classes_functions import BilateralFilter, ImageUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo for Flash/No-Flash photography techniques')
    parser.add_argument('--data_dir', type=str, default='data/camera', 
                        help='Directory containing flash and nonflash subdirectories')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--image_name', type=str, default='cave-flash.jpg', 
                        help='Image filename to process')
    parser.add_argument('--sigma_s_basic', type=float, default=8.0, 
                        help='Spatial sigma for basic bilateral filtering')
    parser.add_argument('--sigma_r_basic', type=float, default=0.1, 
                        help='Range sigma for basic bilateral filtering')
    parser.add_argument('--sigma_s_joint', type=float, default=8.0, 
                        help='Spatial sigma for joint bilateral filtering')
    parser.add_argument('--sigma_r_joint', type=float, default=0.1, 
                        help='Range sigma for joint bilateral filtering')
    parser.add_argument('--epsilon', type=float, default=0.02, 
                        help='Small constant for detail transfer')
    parser.add_argument('--shadow_thresh', type=float, default=0.1, 
                        help='Threshold for shadow detection')
    parser.add_argument('--spec_thresh', type=float, default=0.9, 
                        help='Threshold for specularity detection')
    parser.add_argument('--param_sweep', action='store_true', 
                        help='Perform parameter sweeps')
    
    args = parser.parse_args()
    
    # Create instances of our classes
    bilateral_filter = BilateralFilter()
    image_utils = ImageUtils()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load flash/no-flash image pair
    print(f"Loading images from {args.data_dir}...")
    try:
        ambient_image, flash_image = image_utils.load_flash_no_flash_pair(args.data_dir, args.image_name)
    except Exception as e:
        print(f"Error loading images: {e}")
        # Try with different filename pattern
        base_name = os.path.splitext(args.image_name)[0]
        if '-flash' in base_name:
            noflash_name = args.image_name.replace('-flash', '-ambient')
        else:
            noflash_name = base_name + '-ambient' + os.path.splitext(args.image_name)[1]
            
        try:
            ambient_image, flash_image = image_utils.load_flash_no_flash_pair(
                os.path.dirname(args.data_dir), args.image_name, noflash_name)
            print("Successfully loaded images with alternate naming pattern")
        except:
            print("Could not load images. Please check the file paths and names.")
            exit(1)
    
    print("Images loaded successfully")
    
    # Process images
    print("Processing images...")
    results = bilateral_filter.process_flash_no_flash_pair(
        ambient_image, flash_image,
        sigma_s_basic=args.sigma_s_basic,
        sigma_r_basic=args.sigma_r_basic,
        sigma_s_joint=args.sigma_s_joint,
        sigma_r_joint=args.sigma_r_joint,
        epsilon=args.epsilon,
        shadow_thresh=args.shadow_thresh,
        spec_thresh=args.spec_thresh
    )
    
    # Plot comparison
    print("Generating comparison plots...")
    image_output_dir = os.path.join(args.output_dir, os.path.splitext(args.image_name)[0])
    os.makedirs(image_output_dir, exist_ok=True)
    image_utils.plot_comparison(results, save_dir=image_output_dir)
    
    print(f"Results saved to {image_output_dir}")
    
    # Parameter sweep if requested
    if args.param_sweep:
        print("Performing parameter sweeps...")
        param_sweep_dir = os.path.join(args.output_dir, "param_sweeps")
        os.makedirs(param_sweep_dir, exist_ok=True)
        
        # Sweep spatial sigma for basic bilateral filtering
        sigma_s_values = [2, 4, 8, 16, 32, 64]
        print("Sweeping sigma_s_basic...")
        image_utils.parameter_sweep(
            bilateral_filter, ambient_image, flash_image,
            param_name='sigma_s_basic',
            param_values=sigma_s_values,
            output_dir=os.path.join(param_sweep_dir, "sigma_s_basic")
        )
        
        # Sweep range sigma for basic bilateral filtering
        sigma_r_values = [0.05, 0.1, 0.15, 0.2, 0.25]
        print("Sweeping sigma_r_basic...")
        image_utils.parameter_sweep(
            bilateral_filter, ambient_image, flash_image,
            param_name='sigma_r_basic',
            param_values=sigma_r_values,
            output_dir=os.path.join(param_sweep_dir, "sigma_r_basic")
        )
        
        # Sweep spatial sigma for joint bilateral filtering
        print("Sweeping sigma_s_joint...")
        image_utils.parameter_sweep(
            bilateral_filter, ambient_image, flash_image,
            param_name='sigma_s_joint',
            param_values=sigma_s_values,
            output_dir=os.path.join(param_sweep_dir, "sigma_s_joint")
        )
        
        # Sweep range sigma for joint bilateral filtering
        print("Sweeping sigma_r_joint...")
        image_utils.parameter_sweep(
            bilateral_filter, ambient_image, flash_image,
            param_name='sigma_r_joint',
            param_values=sigma_r_values,
            output_dir=os.path.join(param_sweep_dir, "sigma_r_joint")
        )
        
        # Sweep epsilon for detail transfer
        epsilon_values = [0.005, 0.01, 0.02, 0.05, 0.1]
        print("Sweeping epsilon...")
        image_utils.parameter_sweep(
            bilateral_filter, ambient_image, flash_image,
            param_name='epsilon',
            param_values=epsilon_values,
            output_dir=os.path.join(param_sweep_dir, "epsilon")
        )
        
        print(f"Parameter sweep results saved to {param_sweep_dir}")
    
    print("Demo completed successfully")

if __name__ == "__main__":
    main() 