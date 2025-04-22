"""
Parameter Exploration for Flash/No-Flash Photography
===================================================

This script explores different parameter values for the bilateral filtering techniques
described in the paper by Petschnigg et al.

It allows comparing the effects of different parameter values on the final result,
including:
- Spatial sigma for basic bilateral filtering
- Range sigma for basic bilateral filtering
- Spatial sigma for joint bilateral filtering
- Range sigma for joint bilateral filtering
- Epsilon value for detail transfer
- Shadow and specularity thresholds
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from classes_functions import BilateralFilter, ImageUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parameter exploration for Flash/No-Flash photography')
    parser.add_argument('--data_dir', type=str, default='data/camera', 
                        help='Directory containing flash and nonflash subdirectories')
    parser.add_argument('--output_dir', type=str, default='results/param_exploration', 
                        help='Directory to save results')
    parser.add_argument('--image_name', type=str, default='cave-flash.jpg', 
                        help='Image filename to process')
    
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
        print("Images loaded successfully")
    except Exception as e:
        print(f"Error loading images: {e}")
        exit(1)
    
    # Define parameter ranges to explore
    param_explorations = [
        {
            'name': 'sigma_s_basic',
            'values': [2, 4, 8, 16, 32, 64],
            'title': 'Spatial Sigma for Basic Bilateral Filtering'
        },
        {
            'name': 'sigma_r_basic',
            'values': [0.05, 0.1, 0.15, 0.2, 0.25],
            'title': 'Range Sigma for Basic Bilateral Filtering'
        },
        {
            'name': 'sigma_s_joint',
            'values': [2, 4, 8, 16, 32, 64],
            'title': 'Spatial Sigma for Joint Bilateral Filtering'
        },
        {
            'name': 'sigma_r_joint',
            'values': [0.05, 0.1, 0.15, 0.2, 0.25],
            'title': 'Range Sigma for Joint Bilateral Filtering'
        },
        {
            'name': 'epsilon',
            'values': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            'title': 'Epsilon for Detail Transfer'
        },
        {
            'name': 'shadow_thresh',
            'values': [0.05, 0.1, 0.15, 0.2, 0.25],
            'title': 'Shadow Threshold'
        },
        {
            'name': 'spec_thresh',
            'values': [0.75, 0.8, 0.85, 0.9, 0.95],
            'title': 'Specularity Threshold'
        }
    ]
    
    # Process images with each parameter value
    for param_exp in param_explorations:
        print(f"Exploring parameter: {param_exp['name']}")
        param_dir = os.path.join(args.output_dir, param_exp['name'])
        os.makedirs(param_dir, exist_ok=True)
        
        # Run parameter sweep
        results = image_utils.parameter_sweep(
            bilateral_filter, 
            ambient_image, 
            flash_image,
            param_name=param_exp['name'],
            param_values=param_exp['values'],
            output_dir=param_dir
        )
        
        # Create comparison plot showing each step for each parameter value
        for step in ['a_base', 'a_nr', 'a_detail', 'a_final', 'mask']:
            step_images = [result[step] for result in results]
            step_titles = [f"{param_exp['name']}={value}" for value in param_exp['values']]
            
            plt.figure(figsize=(18, 10))
            plt.suptitle(f"{param_exp['title']} - Effect on {step}")
            
            for i, (img, title) in enumerate(zip(step_images, step_titles)):
                plt.subplot(2, 3, i+1)
                if step == 'mask' or step == 'a_mask':
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(img)
                plt.title(title)
                plt.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(param_dir, f"{step}_comparison.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Results saved to {param_dir}")
    
    # Comparing all algorithmic steps with best parameters
    print("Generating comparison of algorithmic steps...")
    best_params = {
        'sigma_s_basic': 8.0,
        'sigma_r_basic': 0.1,
        'sigma_s_joint': 8.0,
        'sigma_r_joint': 0.1,
        'epsilon': 0.02,
        'shadow_thresh': 0.1,
        'spec_thresh': 0.9
    }
    
    # Process with best parameters
    results = bilateral_filter.process_flash_no_flash_pair(
        ambient_image, flash_image, **best_params
    )
    
    # Plot all steps of the algorithm
    steps_dir = os.path.join(args.output_dir, "algorithm_steps")
    os.makedirs(steps_dir, exist_ok=True)
    image_utils.plot_comparison(results, save_dir=steps_dir)
    
    # Generate individual difference images
    diff_images = []
    diff_titles = []
    
    # ambient vs a_base (basic bilateral filter)
    diff_a_base = image_utils.calculate_difference(results["ambient"], results["a_base"])
    diff_images.append(diff_a_base)
    diff_titles.append("Ambient vs Basic Bilateral")
    
    # a_base vs a_nr (joint bilateral filter)
    diff_base_nr = image_utils.calculate_difference(results["a_base"], results["a_nr"])
    diff_images.append(diff_base_nr)
    diff_titles.append("Basic vs Joint Bilateral")
    
    # a_nr vs a_detail (detail transfer)
    diff_nr_detail = image_utils.calculate_difference(results["a_nr"], results["a_detail"])
    diff_images.append(diff_nr_detail)
    diff_titles.append("Joint Bilateral vs Detail Transfer")
    
    # a_detail vs a_final (shadow/specularity masking)
    diff_detail_final = image_utils.calculate_difference(results["a_detail"], results["a_final"])
    diff_images.append(diff_detail_final)
    diff_titles.append("Detail Transfer vs Final")
    
    # Plot difference images
    image_utils.plot_images(
        diff_images, 
        diff_titles, 
        figsize=(18, 5), 
        rows=1, 
        cols=4, 
        save_path=os.path.join(steps_dir, "algorithm_diff_steps.png")
    )
    
    print(f"Algorithm steps comparison saved to {steps_dir}")
    print("Parameter exploration completed successfully")

if __name__ == "__main__":
    main() 