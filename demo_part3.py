"""
Flash/No-Flash Photography Part 3 Demo
=======================================

This script processes the custom flash/no-flash image pairs from part3 directory:
1. The first pair is processed using bilateral filtering (appropriate for denoising in dimly lit environments)
2. The second pair is processed using gradient domain fusion (appropriate for scenes with mixed matte/specular surfaces)

Both methods are applied with optimal parameters determined from previous experiments.
"""
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from classes_functions import BilateralFilter, GradientDomainProcessor, ImageUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo for Part 3 - Processing custom flash/no-flash image pairs')
    parser.add_argument('--output_dir', type=str, default='results_part3', 
                        help='Directory to save results')
    parser.add_argument('--skip_bilateral', action='store_true',
                        help='Skip bilateral filtering processing')
    parser.add_argument('--skip_gradient', action='store_true',
                        help='Skip gradient domain processing')
    
    args = parser.parse_args()
    
    # Create instances of our classes
    bilateral_filter = BilateralFilter()
    gradient_processor = GradientDomainProcessor()
    image_utils = ImageUtils()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the first pair using bilateral filtering
    if not args.skip_bilateral:
        process_pair_bilateral(
            bilateral_filter=bilateral_filter,
            image_utils=image_utils,
            input_dir='part3/1',
            output_dir=os.path.join(args.output_dir, 'pair1_bilateral')
        )
    
    # Process the second pair using gradient domain processing
    if not args.skip_gradient:
        process_pair_gradient(
            gradient_processor=gradient_processor,
            image_utils=image_utils,
            input_dir='part3/2',
            output_dir=os.path.join(args.output_dir, 'pair2_gradient')
        )
    
    print("Part 3 demo completed successfully")

def process_pair_bilateral(bilateral_filter, image_utils, input_dir, output_dir):
    """
    Process a flash/no-flash image pair using bilateral filtering
    
    Args:
        bilateral_filter: BilateralFilter instance
        image_utils: ImageUtils instance
        input_dir: Directory containing flash and nonflash subdirectories
        output_dir: Directory to save results
    """
    print(f"\nProcessing image pair from {input_dir} using bilateral filtering...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first image in each directory
    flash_dir = os.path.join(input_dir, 'flash')
    nonflash_dir = os.path.join(input_dir, 'nonflash')
    
    flash_files = [f for f in os.listdir(flash_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    nonflash_files = [f for f in os.listdir(nonflash_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    if not flash_files or not nonflash_files:
        print("Error: Could not find image files in flash or nonflash directories")
        return
    
    flash_path = os.path.join(flash_dir, flash_files[0])
    nonflash_path = os.path.join(nonflash_dir, nonflash_files[0])
    
    print(f"Processing flash image: {flash_path}")
    print(f"Processing nonflash image: {nonflash_path}")
    
    # Load images
    flash_image = image_utils.load_image(flash_path)
    ambient_image = image_utils.load_image(nonflash_path)
    
    # Process with optimal parameters determined from previous experiments
    results = bilateral_filter.process_flash_no_flash_pair(
        ambient_image=ambient_image,
        flash_image=flash_image,
        sigma_s_basic=8.0,
        sigma_r_basic=0.1,
        sigma_s_joint=8.0,
        sigma_r_joint=0.1,
        epsilon=0.02,
        shadow_thresh=0.1,
        spec_thresh=0.9
    )
    
    # Plot comparison and save individual images
    image_utils.plot_comparison(results, save_dir=output_dir)
    
    # Create a side-by-side comparison of ambient, flash, and final result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(ambient_image)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(flash_image)
    plt.title("Flash")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(results["a_final"])
    plt.title("Bilateral Filtering Result")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_bilateral.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Bilateral filtering results saved to {output_dir}")

def process_pair_gradient(gradient_processor, image_utils, input_dir, output_dir):
    """
    Process a flash/no-flash image pair using gradient domain processing
    
    Args:
        gradient_processor: GradientDomainProcessor instance
        image_utils: ImageUtils instance
        input_dir: Directory containing flash and nonflash subdirectories
        output_dir: Directory to save results
    """
    print(f"\nProcessing image pair from {input_dir} using gradient domain processing...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first image in each directory
    flash_dir = os.path.join(input_dir, 'flash')
    nonflash_dir = os.path.join(input_dir, 'nonflash')
    
    flash_files = [f for f in os.listdir(flash_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    nonflash_files = [f for f in os.listdir(nonflash_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    if not flash_files or not nonflash_files:
        print("Error: Could not find image files in flash or nonflash directories")
        return
    
    flash_path = os.path.join(flash_dir, flash_files[0])
    nonflash_path = os.path.join(nonflash_dir, nonflash_files[0])
    
    print(f"Processing flash image: {flash_path}")
    print(f"Processing nonflash image: {nonflash_path}")
    
    # Load images
    flash_image = image_utils.load_image(flash_path)
    ambient_image = image_utils.load_image(nonflash_path)
    
    # Process with optimal parameters determined from previous experiments
    results = gradient_processor.process_flash_no_flash_pair(
        ambient_image=ambient_image,
        flash_image=flash_image,
        sigma=5.0,
        tau_s=0.12,
        boundary_type="ambient",
        init_type="average",
        epsilon=1e-6,
        max_iterations=1000
    )
    
    # Plot comparison
    gradient_processor.plot_results(results, save_path=os.path.join(output_dir, "results.png"))
    
    # Generate gradient field visualization
    gradient_results = gradient_processor.poisson_solver.generate_fused_gradient(
        ambient_image, flash_image, sigma=5.0, tau_s=0.12
    )
    gradient_processor.plot_gradient_fields(
        ambient_image, flash_image, gradient_results,
        save_path=os.path.join(output_dir, "gradient_fields.png")
    )
    
    # Save individual images
    for name, img in results.items():
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Save grayscale images with viridis colormap
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_colormap.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save raw image
        if img.max() <= 1.0:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = img.astype(np.uint8)
            
        if len(img.shape) == 2:
            img_save = cv2.cvtColor(img_save, cv2.COLOR_GRAY2BGR)
            
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    
    # Create a side-by-side comparison of ambient, flash, and final result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(ambient_image)
    plt.title("Ambient (No Flash)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(flash_image)
    plt.title("Flash")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(results["fused_image"])
    plt.title("Gradient Domain Result")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_gradient.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient domain processing results saved to {output_dir}")

if __name__ == "__main__":
    main() 