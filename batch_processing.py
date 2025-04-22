"""
Batch Processing for Flash/No-Flash Photography
===============================================

This script processes all flash/no-flash image pairs in a dataset using the bilateral filtering 
techniques described in the paper by Petschnigg et al.
"""
import os
import argparse
import numpy as np
from classes_functions import BilateralFilter, ImageUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch processing for Flash/No-Flash photography')
    parser.add_argument('--data_dir', type=str, default='data/camera', 
                        help='Directory containing flash and nonflash subdirectories')
    parser.add_argument('--output_dir', type=str, default='results/batch', 
                        help='Directory to save results')
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
    
    args = parser.parse_args()
    
    # Create instances of our classes
    bilateral_filter = BilateralFilter()
    image_utils = ImageUtils()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all images in the dataset
    print(f"Processing all images in {args.data_dir}...")
    try:
        # Try to find images in the nonflash directory
        nonflash_dir = os.path.join(args.data_dir, 'nonflash')
        if os.path.exists(nonflash_dir):
            image_names = [f for f in os.listdir(nonflash_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if not image_names:
                print("No images found in nonflash directory")
                return
                
            print(f"Found {len(image_names)} image pairs to process")
            
            # Process each image pair
            results = image_utils.batch_process_dataset(
                bilateral_filter, 
                args.data_dir, 
                args.output_dir,
                image_names=image_names,
                sigma_s_basic=args.sigma_s_basic,
                sigma_r_basic=args.sigma_r_basic,
                sigma_s_joint=args.sigma_s_joint,
                sigma_r_joint=args.sigma_r_joint,
                epsilon=args.epsilon
            )
            
            print(f"All images processed successfully. Results saved to {args.output_dir}")
            
        else:
            print(f"Could not find nonflash directory at {nonflash_dir}")
            
    except Exception as e:
        print(f"Error processing images: {e}")

if __name__ == "__main__":
    main() 