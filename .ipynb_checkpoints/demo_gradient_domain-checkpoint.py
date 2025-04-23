"""
Flash/No-Flash Photography Demo - Gradient Domain Processing
===========================================================

This script demonstrates the implementation of gradient-domain processing for flash/no-flash
photography as described in the paper by Agrawal et al.

The implementation consists of two main parts:
1. Differentiate and re-integrate an image using a Poisson solver with conjugate gradient descent
2. Create a fused gradient field from flash and ambient images, then integrate it to get the final result
"""
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from classes_functions import GradientDomainProcessor, ImageUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo for Gradient-Domain Flash/No-Flash photography techniques')
    parser.add_argument('--data_dir', type=str, default='data/face', 
                        help='Directory containing flash and nonflash subdirectories')
    parser.add_argument('--output_dir', type=str, default='results_gradient', 
                        help='Directory to save results')
    parser.add_argument('--image_name', type=str, default='face-flash.jpg', 
                        help='Image filename to process')
    parser.add_argument('--test_poisson', action='store_true',
                        help='Test Poisson solver by differentiating and re-integrating an image')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Parameter for saturation weight calculation')
    parser.add_argument('--tau_s', type=float, default=0.12,
                        help='Threshold for saturation weight calculation')
    parser.add_argument('--boundary_type', type=str, default='ambient',
                        choices=['ambient', 'flash', 'average'],
                        help='Type of boundary conditions for Poisson integration')
    parser.add_argument('--init_type', type=str, default='average',
                        choices=['ambient', 'flash', 'average', 'zero'],
                        help='Type of initialization for Poisson integration')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        help='Convergence parameter for conjugate gradient descent')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum number of iterations for conjugate gradient descent')
    
    args = parser.parse_args()
    
    # Create instances of our classes
    processor = GradientDomainProcessor()
    image_utils = ImageUtils()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test Poisson solver if requested
    if args.test_poisson:
        test_poisson_solver(processor, image_utils, args)
    
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
    print(f"Processing flash/no-flash pair using gradient-domain processing...")
    start_time = time.time()
    results = processor.process_flash_no_flash_pair(
        ambient_image, flash_image,
        sigma=args.sigma,
        tau_s=args.tau_s,
        boundary_type=args.boundary_type,
        init_type=args.init_type,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations
    )
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Save results
    image_output_dir = os.path.join(args.output_dir, os.path.splitext(args.image_name)[0])
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Plot and save results
    processor.plot_results(results, save_path=os.path.join(image_output_dir, "results.png"))
    
    # Save individual images
    for name, img in results.items():
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Save grayscale images with viridis colormap
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(image_output_dir, f"{name}_colormap.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save raw image
        if img.max() <= 1.0:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = img.astype(np.uint8)
            
        if len(img.shape) == 2:
            img_save = cv2.cvtColor(img_save, cv2.COLOR_GRAY2BGR)
            
        cv2.imwrite(os.path.join(image_output_dir, f"{name}.png"), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    
    # Generate gradient field visualization
    print("Generating gradient field visualization...")
    gradient_results = processor.poisson_solver.generate_fused_gradient(
        ambient_image, flash_image, args.sigma, args.tau_s
    )
    processor.plot_gradient_fields(
        ambient_image, flash_image, gradient_results,
        save_path=os.path.join(image_output_dir, "gradient_fields.png")
    )
    
    # Try different parameter combinations
    if args.boundary_type == 'ambient' and args.init_type == 'average':
        print("Trying different parameter combinations...")
        
        # Try different sigma values
        sigma_values = [1.0, 5.0, 10.0]
        for sigma in sigma_values:
            if sigma == args.sigma:
                continue
                
            print(f"Processing with sigma={sigma}...")
            results_sigma = processor.process_flash_no_flash_pair(
                ambient_image, flash_image,
                sigma=sigma,
                tau_s=args.tau_s,
                boundary_type=args.boundary_type,
                init_type=args.init_type,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations
            )
            
            processor.plot_results(
                results_sigma, 
                save_path=os.path.join(image_output_dir, f"results_sigma_{sigma}.png")
            )
        
        # Try different tau_s values
        tau_s_values = [0.05, 0.12, 0.2]
        for tau_s in tau_s_values:
            if tau_s == args.tau_s:
                continue
                
            print(f"Processing with tau_s={tau_s}...")
            results_tau_s = processor.process_flash_no_flash_pair(
                ambient_image, flash_image,
                sigma=args.sigma,
                tau_s=tau_s,
                boundary_type=args.boundary_type,
                init_type=args.init_type,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations
            )
            
            processor.plot_results(
                results_tau_s, 
                save_path=os.path.join(image_output_dir, f"results_tau_s_{tau_s}.png")
            )
    
    # Create a parameter comparison figure
    create_parameter_comparison(processor, ambient_image, flash_image, image_output_dir)
    
    print(f"Results saved to {image_output_dir}")
    print("Demo completed successfully")

def test_poisson_solver(processor, image_utils, args):
    """
    Test the Poisson solver by differentiating and re-integrating an image
    """
    print("Testing Poisson solver with differentiate and re-integrate...")
    
    # Use ambient image for testing
    print(f"Loading test image from {args.data_dir}...")
    try:
        ambient_image, _ = image_utils.load_flash_no_flash_pair(args.data_dir, args.image_name)
    except Exception as e:
        print(f"Error loading test image: {e}")
        return
    
    # Create output directory for test results
    test_output_dir = os.path.join(args.output_dir, "poisson_test")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Test for different epsilon values
    epsilon_values = [1e-3, 1e-6, 1e-9]
    for epsilon in epsilon_values:
        print(f"Testing with epsilon={epsilon}...")
        
        # Run test
        start_time = time.time()
        test_results = processor.test_differentiate_reintegrate(
            ambient_image, epsilon=epsilon, max_iterations=args.max_iterations
        )
        test_time = time.time() - start_time
        
        print(f"Test completed in {test_time:.2f} seconds")
        print(f"Mean Error: {test_results['mean_error']}")
        print(f"Max Error: {test_results['max_error']}")
        
        # Plot and save test results
        processor.plot_test_results(
            test_results, 
            save_path=os.path.join(test_output_dir, f"test_results_epsilon_{epsilon}.png")
        )
    
    print("Poisson solver test completed")

def create_parameter_comparison(processor, ambient_image, flash_image, output_dir):
    """
    Create a comparison of results with different parameter combinations
    """
    print("Creating parameter comparison figure...")
    
    # Parameter combinations to test
    param_combinations = [
        {'sigma': 5.0, 'tau_s': 0.12, 'boundary_type': 'ambient', 'init_type': 'average'},
        {'sigma': 5.0, 'tau_s': 0.12, 'boundary_type': 'flash', 'init_type': 'average'},
        {'sigma': 5.0, 'tau_s': 0.12, 'boundary_type': 'average', 'init_type': 'average'},
        {'sigma': 5.0, 'tau_s': 0.12, 'boundary_type': 'ambient', 'init_type': 'flash'},
        {'sigma': 5.0, 'tau_s': 0.12, 'boundary_type': 'ambient', 'init_type': 'zero'},
        {'sigma': 1.0, 'tau_s': 0.12, 'boundary_type': 'ambient', 'init_type': 'average'}
    ]
    
    # Process each combination
    results_list = []
    labels = []
    
    for i, params in enumerate(param_combinations):
        print(f"Processing combination {i+1}/{len(param_combinations)}...")
        
        # Process images with current parameters
        results = processor.process_flash_no_flash_pair(
            ambient_image, flash_image,
            sigma=params['sigma'],
            tau_s=params['tau_s'],
            boundary_type=params['boundary_type'],
            init_type=params['init_type'],
            epsilon=1e-6,
            max_iterations=1000
        )
        
        # Store result
        results_list.append(results["fused_image"])
        
        # Create label
        label = f"σ={params['sigma']}, τ={params['tau_s']}\n"
        label += f"B={params['boundary_type'][:3]}, I={params['init_type'][:3]}"
        labels.append(label)
    
    # Create comparison figure
    plt.figure(figsize=(15, 10))
    
    # Plot ambient and flash images
    plt.subplot(2, 4, 1)
    plt.imshow(ambient_image)
    plt.title("Ambient Image")
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(flash_image)
    plt.title("Flash Image")
    plt.axis('off')
    
    # Plot results for each parameter combination
    for i, (result, label) in enumerate(zip(results_list, labels)):
        plt.subplot(2, 4, i+3)
        plt.imshow(result)
        plt.title(label)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Parameter comparison figure created")

if __name__ == "__main__":
    main() 