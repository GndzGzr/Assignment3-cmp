"""
Batch Processing for Flash/No-Flash Photography with GPU Acceleration
====================================================================

This script processes all flash/no-flash image pairs in a dataset using the bilateral filtering 
techniques described in the paper by Petschnigg et al. with GPU acceleration.
"""
import os
import argparse
import numpy as np
import time
from classes_functions import BilateralFilterGPU, ImageUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch processing for Flash/No-Flash photography with GPU acceleration')
    parser.add_argument('--data_dir', type=str, default='data/camera', 
                        help='Directory containing flash and nonflash subdirectories')
    parser.add_argument('--output_dir', type=str, default='results_gpu/batch', 
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
    parser.add_argument('--no_gpu', action='store_true', 
                        help='Disable GPU acceleration even if available')
    parser.add_argument('--compare_cpu', action='store_true',
                        help='Compare performance with CPU implementation')
    
    args = parser.parse_args()
    
    # Create instances of our classes
    bilateral_filter_gpu = BilateralFilterGPU(use_gpu=not args.no_gpu)
    image_utils = ImageUtils()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all images in the dataset
    print(f"Processing all images in {args.data_dir} with GPU acceleration...")
    try:
        # Try to find images in the nonflash directory
        nonflash_dir = os.path.join(args.data_dir, 'nonflash')
        if os.path.exists(nonflash_dir):
            image_names = [f for f in os.listdir(nonflash_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if not image_names:
                print("No images found in nonflash directory")
                return
                
            print(f"Found {len(image_names)} image pairs to process")
            
            # Create a performance log file
            perf_log_path = os.path.join(args.output_dir, "performance_log.txt")
            with open(perf_log_path, 'w') as perf_log:
                perf_log.write("Performance comparison for GPU vs CPU processing\n")
                perf_log.write("================================================\n\n")
                
                total_gpu_time = 0
                total_cpu_time = 0
                
                # Process each image pair
                for img_name in image_names:
                    print(f"Processing {img_name}...")
                    
                    # Load image pair
                    ambient_image, flash_image = image_utils.load_flash_no_flash_pair(args.data_dir, img_name)
                    
                    # Process images with GPU
                    start_time = time.time()
                    result_gpu = bilateral_filter_gpu.process_flash_no_flash_pair(
                        ambient_image, flash_image,
                        sigma_s_basic=args.sigma_s_basic,
                        sigma_r_basic=args.sigma_r_basic,
                        sigma_s_joint=args.sigma_s_joint,
                        sigma_r_joint=args.sigma_r_joint,
                        epsilon=args.epsilon
                    )
                    gpu_time = time.time() - start_time
                    total_gpu_time += gpu_time
                    
                    # Save GPU results
                    img_output_dir = os.path.join(args.output_dir, os.path.splitext(img_name)[0])
                    os.makedirs(img_output_dir, exist_ok=True)
                    
                    # Plot comparison
                    image_utils.plot_comparison(result_gpu, save_dir=img_output_dir)
                    
                    # Process with CPU if comparison requested
                    if args.compare_cpu:
                        from classes_functions import BilateralFilter
                        bilateral_filter_cpu = BilateralFilter()
                        
                        start_time = time.time()
                        result_cpu = bilateral_filter_cpu.process_flash_no_flash_pair(
                            ambient_image, flash_image,
                            sigma_s_basic=args.sigma_s_basic,
                            sigma_r_basic=args.sigma_r_basic,
                            sigma_s_joint=args.sigma_s_joint,
                            sigma_r_joint=args.sigma_r_joint,
                            epsilon=args.epsilon
                        )
                        cpu_time = time.time() - start_time
                        total_cpu_time += cpu_time
                        
                        # Calculate PSNR between CPU and GPU results
                        psnr = image_utils.calculate_psnr(result_cpu["a_final"], result_gpu["a_final"])
                        
                        # Write to performance log
                        perf_log.write(f"Image: {img_name}\n")
                        perf_log.write(f"  GPU time: {gpu_time:.2f} seconds\n")
                        perf_log.write(f"  CPU time: {cpu_time:.2f} seconds\n")
                        perf_log.write(f"  Speedup: {cpu_time / gpu_time:.2f}x\n")
                        perf_log.write(f"  PSNR: {psnr:.2f} dB\n\n")
                        
                        # Save CPU vs GPU comparison image
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        plt.imshow(result_cpu["a_final"])
                        plt.title("CPU Result")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(result_gpu["a_final"])
                        plt.title("GPU Result")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        diff = np.abs(result_cpu["a_final"] - result_gpu["a_final"]) * 10  # Scale difference for visualization
                        plt.imshow(diff)
                        plt.title(f"Difference (x10) - PSNR: {psnr:.2f} dB")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(img_output_dir, "cpu_vs_gpu.png"), dpi=150, bbox_inches='tight')
                        plt.close()
                    else:
                        # Just log GPU time
                        perf_log.write(f"Image: {img_name}\n")
                        perf_log.write(f"  GPU time: {gpu_time:.2f} seconds\n\n")
                
                # Write summary
                perf_log.write("\nSummary\n=======\n")
                perf_log.write(f"Total images processed: {len(image_names)}\n")
                perf_log.write(f"Total GPU processing time: {total_gpu_time:.2f} seconds\n")
                if args.compare_cpu:
                    perf_log.write(f"Total CPU processing time: {total_cpu_time:.2f} seconds\n")
                    perf_log.write(f"Average speedup: {total_cpu_time / total_gpu_time:.2f}x\n")
            
            print(f"All images processed successfully. Results saved to {args.output_dir}")
            print(f"Performance log saved to {perf_log_path}")
            
        else:
            print(f"Could not find nonflash directory at {nonflash_dir}")
            
    except Exception as e:
        print(f"Error processing images: {e}")

if __name__ == "__main__":
    main() 