import numpy as np
import cv2
import matplotlib.pyplot as plt
from .poisson_solver import PoissonSolver

class GradientDomainProcessor:
    def __init__(self):
        """
        Initialize the gradient domain processor
        """
        self.poisson_solver = PoissonSolver()
    
    def process_flash_no_flash_pair(self, ambient_image, flash_image, 
                                   sigma=5.0, tau_s=0.12,
                                   boundary_type="ambient", init_type="average",
                                   epsilon=1e-6, max_iterations=1000):
        """
        Process a flash/no-flash image pair using gradient-domain processing
        
        Args:
            ambient_image: Image taken under ambient lighting (a)
            flash_image: Image taken with flash (Φ')
            sigma: Parameter for saturation weight calculation
            tau_s: Threshold for saturation weight calculation
            boundary_type: Type of boundary conditions ("ambient", "flash", or "average")
            init_type: Type of initialization ("ambient", "flash", "average", or "zero")
            epsilon: Convergence parameter for CGD
            max_iterations: Maximum number of iterations for CGD
            
        Returns:
            Dictionary containing processed image and intermediate results
        """
        # Ensure images are float and normalized
        if ambient_image.max() > 1.0:
            ambient_image = ambient_image.astype(np.float32) / 255.0
        if flash_image.max() > 1.0:
            flash_image = flash_image.astype(np.float32) / 255.0
        
        # Step 1: Generate fused gradient field
        print("Generating fused gradient field...")
        gradient_results = self.poisson_solver.generate_fused_gradient(
            ambient_image, flash_image, sigma, tau_s
        )
        
        # Step 2: Integrate the fused gradient field
        print("Integrating fused gradient field...")
        fused_image = self.poisson_solver.integrate_fused_gradient(
            gradient_results["fused_grad_x"],
            gradient_results["fused_grad_y"],
            ambient_image,
            flash_image,
            boundary_type,
            init_type,
            epsilon,
            max_iterations
        )
        
        # Ensure output is in valid range
        fused_image = np.clip(fused_image, 0.0, 1.0)
        
        # Return results including intermediate products
        return {
            "ambient": ambient_image,
            "flash": flash_image,
            "coherency_map": gradient_results["coherency_map"],
            "saturation_weight": gradient_results["saturation_weight"],
            "fused_image": fused_image
        }
    
    def test_differentiate_reintegrate(self, image, epsilon=1e-6, max_iterations=1000):
        """
        Test function to differentiate and then reintegrate an image
        
        Args:
            image: Input image
            epsilon: Convergence parameter for CGD
            max_iterations: Maximum number of iterations for CGD
            
        Returns:
            Dictionary containing original and reintegrated images
        """
        # Ensure image is float and normalized
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Differentiate and reintegrate
        print("Testing differentiate and reintegrate...")
        reintegrated = self.poisson_solver.differentiate_and_reintegrate(
            image, epsilon, max_iterations
        )
        
        # Compute error
        error = np.abs(image - reintegrated)
        mean_error = np.mean(error)
        max_error = np.max(error)
        
        print(f"Mean Error: {mean_error}")
        print(f"Max Error: {max_error}")
        
        return {
            "original": image,
            "reintegrated": reintegrated,
            "error": error,
            "mean_error": mean_error,
            "max_error": max_error
        }
    
    def visualize_gradient_field(self, grad_x, grad_y, step=16, scale=1.0, color=(0, 0, 255)):
        """
        Visualize a gradient field with arrows
        
        Args:
            grad_x: X component of gradient field
            grad_y: Y component of gradient field
            step: Sampling step for visualization (to avoid too many arrows)
            scale: Scale factor for arrow length
            color: Color of the arrows (B, G, R)
            
        Returns:
            Visualization image
        """
        # Create a blank image
        height, width = grad_x.shape[:2]
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sample points for visualization
        y, x = np.mgrid[step//2:height:step, step//2:width:step].reshape(2, -1).astype(int)
        
        # Sample gradient values
        if len(grad_x.shape) == 3:  # Color image, use first channel
            fx = grad_x[y, x, 0]
            fy = grad_y[y, x, 0]
        else:  # Grayscale image
            fx = grad_x[y, x]
            fy = grad_y[y, x]
        
        # Calculate arrow endpoints
        lines = np.vstack([x, y, x + fx * scale, y + fy * scale]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        
        # Draw arrows
        cv2.polylines(vis, lines, False, color)
        
        # Draw arrow heads
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x2, y2), 1, color, -1)
        
        return vis
    
    def plot_results(self, results, save_path=None):
        """
        Plot results of flash/no-flash gradient-domain processing
        
        Args:
            results: Dictionary of results from process_flash_no_flash_pair
            save_path: Path to save the figure (if None, figure is displayed)
        """
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot ambient and flash images
        plt.subplot(2, 3, 1)
        plt.imshow(results["ambient"])
        plt.title("Ambient Image")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(results["flash"])
        plt.title("Flash Image")
        plt.axis('off')
        
        # Plot coherency map
        plt.subplot(2, 3, 3)
        plt.imshow(results["coherency_map"], cmap='viridis')
        plt.title("Coherency Map")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Plot saturation weight
        plt.subplot(2, 3, 4)
        plt.imshow(results["saturation_weight"], cmap='viridis')
        plt.title("Saturation Weight")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Plot fused image
        plt.subplot(2, 3, 5)
        plt.imshow(results["fused_image"])
        plt.title("Fused Image")
        plt.axis('off')
        
        # Plot difference image
        diff = np.abs(results["ambient"] - results["fused_image"])
        plt.subplot(2, 3, 6)
        plt.imshow(diff * 5)  # Scale for better visibility
        plt.title("Difference (Ambient vs Fused) × 5")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_test_results(self, results, save_path=None):
        """
        Plot results of differentiate-reintegrate test
        
        Args:
            results: Dictionary of results from test_differentiate_reintegrate
            save_path: Path to save the figure (if None, figure is displayed)
        """
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(results["original"])
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot reintegrated image
        plt.subplot(1, 3, 2)
        plt.imshow(results["reintegrated"])
        plt.title("Reintegrated Image")
        plt.axis('off')
        
        # Plot error image
        plt.subplot(1, 3, 3)
        plt.imshow(results["error"] * 10)  # Scale for better visibility
        plt.title(f"Error × 10\nMean: {results['mean_error']:.6f}, Max: {results['max_error']:.6f}")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_gradient_fields(self, ambient_image, flash_image, gradient_results, save_path=None):
        """
        Plot gradient fields for visualization
        
        Args:
            ambient_image: Ambient image
            flash_image: Flash image
            gradient_results: Dictionary containing gradient fields from generate_fused_gradient
            save_path: Path to save the figure (if None, figure is displayed)
        """
        # Visualize gradient fields
        ambient_grad_vis = self.visualize_gradient_field(
            gradient_results["grad_a_x"], 
            gradient_results["grad_a_y"],
            step=20, scale=2.0, color=(0, 255, 0)
        )
        
        flash_grad_vis = self.visualize_gradient_field(
            gradient_results["grad_f_x"], 
            gradient_results["grad_f_y"],
            step=20, scale=2.0, color=(0, 0, 255)
        )
        
        fused_grad_vis = self.visualize_gradient_field(
            gradient_results["fused_grad_x"], 
            gradient_results["fused_grad_y"],
            step=20, scale=2.0, color=(255, 0, 0)
        )
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot ambient image and its gradient
        plt.subplot(2, 3, 1)
        plt.imshow(ambient_image)
        plt.title("Ambient Image")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(ambient_grad_vis)
        plt.title("Ambient Gradient Field")
        plt.axis('off')
        
        # Plot flash image and its gradient
        plt.subplot(2, 3, 3)
        plt.imshow(flash_image)
        plt.title("Flash Image")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(flash_grad_vis)
        plt.title("Flash Gradient Field")
        plt.axis('off')
        
        # Plot coherency map and fused gradient
        plt.subplot(2, 3, 5)
        plt.imshow(gradient_results["coherency_map"], cmap='viridis')
        plt.title("Coherency Map")
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(fused_grad_vis)
        plt.title("Fused Gradient Field")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 