import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, linalg
from scipy.ndimage import gaussian_filter


class ImageUtils:
    """Utility class for handling image loading, saving, and visualization."""
    
    def __init__(self):
        pass
    
    def load_flash_no_flash_pair(self, directory, flash_filename, noflash_filename):
        """
        Load a flash/no-flash image pair from the specified directory.
        
        Args:
            directory: Path to the directory containing the images
            flash_filename: Filename of the flash image
            noflash_filename: Filename of the no-flash (ambient) image
            
        Returns:
            tuple: (ambient_image, flash_image) normalized to [0, 1] range
        """
        flash_path = os.path.join(directory, flash_filename)
        noflash_path = os.path.join(directory, noflash_filename)
        
        # Check if files exist
        if not os.path.exists(flash_path):
            raise FileNotFoundError(f"Flash image not found: {flash_path}")
        if not os.path.exists(noflash_path):
            raise FileNotFoundError(f"No-flash image not found: {noflash_path}")
        
        # Load images
        flash_img = cv2.imread(flash_path)
        noflash_img = cv2.imread(noflash_path)
        
        # Convert from BGR to RGB
        flash_img = cv2.cvtColor(flash_img, cv2.COLOR_BGR2RGB)
        noflash_img = cv2.cvtColor(noflash_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] range
        flash_img = flash_img.astype(np.float32) / 255.0
        noflash_img = noflash_img.astype(np.float32) / 255.0
        
        # Ensure images have the same shape
        if flash_img.shape != noflash_img.shape:
            raise ValueError(f"Image dimensions do not match: {flash_img.shape} vs {noflash_img.shape}")
        
        return noflash_img, flash_img
    
    def save_image(self, image, filename):
        """Save an image to the specified filename."""
        # Convert to uint8 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the image
        cv2.imwrite(filename, image_bgr)
    
    def plot_comparison(self, results, save_dir=None):
        """
        Plot comparison of bilateral filtering results.
        
        Args:
            results: Dictionary containing the results of bilateral filtering
            save_dir: Directory to save the output images (optional)
        """
        plt.figure(figsize=(15, 10))
        
        # Original images
        plt.subplot(2, 3, 1)
        plt.imshow(results["a_orig"])
        plt.title("Ambient (No Flash)")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(results["f_orig"])
        plt.title("Flash")
        plt.axis('off')
        
        # Base image
        plt.subplot(2, 3, 3)
        plt.imshow(results["a_base"])
        plt.title("Base (A')")
        plt.axis('off')
        
        # Detail layer
        plt.subplot(2, 3, 4)
        # Scale detail layer for better visibility
        detail = np.clip(results["a_detail"] + 0.5, 0, 1)
        plt.imshow(detail)
        plt.title("Detail Layer")
        plt.axis('off')
        
        # Noise reduction
        plt.subplot(2, 3, 5)
        plt.imshow(results["a_nr"])
        plt.title("Noise Reduction (ANR)")
        plt.axis('off')
        
        # Final result
        plt.subplot(2, 3, 6)
        plt.imshow(results["a_final"])
        plt.title("Final Result")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the figure if a directory is provided
        if save_dir:
            plt.savefig(os.path.join(save_dir, "bilateral_results.png"), dpi=150, bbox_inches='tight')
        
        plt.show()


class BilateralFilter:
    """
    Implementation of bilateral filtering techniques for flash/no-flash photography.
    Based on Petschnigg et al. "Digital Photography with Flash and No-Flash Image Pairs".
    """
    
    def __init__(self):
        pass
    
    def bilateral_filter(self, image, sigma_s, sigma_r):
        """
        Apply bilateral filter to an image.
        
        Args:
            image: Input image (HxWxC)
            sigma_s: Spatial standard deviation
            sigma_r: Range standard deviation
            
        Returns:
            Filtered image
        """
        # Process each channel separately
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = cv2.bilateralFilter(
                image[:, :, c], 
                0,  # Diameter of each pixel neighborhood (0 = auto)
                sigma_r,  # Filter sigma in the color space
                sigma_s   # Filter sigma in the coordinate space
            )
        return result
    
    def joint_bilateral_filter(self, image, guide, sigma_s, sigma_r):
        """
        Apply joint bilateral filter to an image.
        
        Args:
            image: Input image to be filtered (HxWxC)
            guide: Guide image used for edge-preserving filtering (HxWxC)
            sigma_s: Spatial standard deviation
            sigma_r: Range standard deviation
            
        Returns:
            Filtered image
        """
        # Convert images to grayscale for guidance
        if guide.shape[2] == 3:
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        else:
            guide_gray = guide[:, :, 0]
        
        # Process each channel separately
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = cv2.ximgproc.jointBilateralFilter(
                guide_gray, 
                image[:, :, c], 
                -1,  # Diameter of each pixel neighborhood (-1 = auto)
                sigma_r, 
                sigma_s
            )
        
        return result
    
    def detect_flash_shadows_specularities(self, ambient, flash, shadow_thresh=0.1, spec_thresh=0.9):
        """
        Detect shadow and specularity masks from flash/no-flash pair.
        
        Args:
            ambient: Ambient (no-flash) image
            flash: Flash image
            shadow_thresh: Threshold for shadow detection
            spec_thresh: Threshold for specularity detection
            
        Returns:
            tuple: (shadow_mask, specularity_mask)
        """
        # Convert to grayscale
        if ambient.shape[2] == 3:
            ambient_gray = cv2.cvtColor(ambient, cv2.COLOR_RGB2GRAY)
            flash_gray = cv2.cvtColor(flash, cv2.COLOR_RGB2GRAY)
        else:
            ambient_gray = ambient[:, :, 0]
            flash_gray = flash[:, :, 0]
        
        # Compute ratio (flash / no-flash)
        ratio = np.zeros_like(ambient_gray)
        mask = ambient_gray > 0.02  # Avoid division by zero
        ratio[mask] = flash_gray[mask] / ambient_gray[mask]
        
        # Normalize ratio
        ratio = ratio / np.median(ratio[mask])
        
        # Apply Gaussian blur to reduce noise
        ratio = cv2.GaussianBlur(ratio, (0, 0), 2.0)
        
        # Detect shadows and specularities
        shadow_mask = ratio < shadow_thresh
        spec_mask = ratio > spec_thresh
        
        return shadow_mask, spec_mask
    
    def process_flash_no_flash_pair(self, ambient, flash, 
                                    sigma_s_basic=8.0, sigma_r_basic=0.1,
                                    sigma_s_joint=8.0, sigma_r_joint=0.1,
                                    epsilon=0.02, shadow_thresh=0.1, spec_thresh=0.9):
        """
        Process a flash/no-flash image pair using bilateral filtering techniques.
        
        Args:
            ambient: Ambient (no-flash) image (HxWxC)
            flash: Flash image (HxWxC)
            sigma_s_basic: Spatial sigma for basic bilateral filter
            sigma_r_basic: Range sigma for basic bilateral filter
            sigma_s_joint: Spatial sigma for joint bilateral filter
            sigma_r_joint: Range sigma for joint bilateral filter
            epsilon: Detail layer strength
            shadow_thresh: Threshold for shadow detection
            spec_thresh: Threshold for specularity detection
            
        Returns:
            Dictionary containing all intermediate and final results
        """
        # Store original images
        a_orig = ambient.copy()
        f_orig = flash.copy()
        
        # Step 1: Compute the base layer (A') using bilateral filter
        a_base = self.bilateral_filter(ambient, sigma_s_basic, sigma_r_basic)
        
        # Step 2: Compute the flash detail layer (F - F')
        f_base = self.bilateral_filter(flash, sigma_s_basic, sigma_r_basic)
        f_detail = flash - f_base
        
        # Step 3: Apply noise reduction using joint bilateral filter
        a_nr = self.joint_bilateral_filter(ambient, flash, sigma_s_joint, sigma_r_joint)
        
        # Step 4: Compute the detail layer for the final image
        # a_detail = epsilon * f_detail
        
        # Use the flash image to compute a better detail scaling factor
        # that adjusts based on local intensity
        detail_scale = np.maximum(0.05, np.mean(flash, axis=2))
        detail_scale = np.stack([detail_scale] * 3, axis=2)
        a_detail = f_detail * detail_scale * epsilon
        
        # Step 5: Detect flash shadows and specularities
        shadow_mask, spec_mask = self.detect_flash_shadows_specularities(
            ambient, flash, shadow_thresh, spec_thresh
        )
        
        # Create 3-channel masks
        shadow_mask_3c = np.stack([shadow_mask] * 3, axis=2)
        spec_mask_3c = np.stack([spec_mask] * 3, axis=2)
        
        # Step 6: Combine results with special handling for shadows and specularities
        a_final = a_base + a_detail
        
        # In shadow regions, use the noise-reduced ambient image
        a_final = np.where(shadow_mask_3c, a_nr, a_final)
        
        # In specular regions, use the base ambient image
        a_final = np.where(spec_mask_3c, a_base, a_final)
        
        # Clip values to valid range
        a_final = np.clip(a_final, 0, 1)
        
        # Return all intermediate results and the final image
        return {
            "a_orig": a_orig,
            "f_orig": f_orig,
            "a_base": a_base,
            "f_base": f_base,
            "f_detail": f_detail,
            "a_nr": a_nr,
            "a_detail": a_detail,
            "shadow_mask": shadow_mask,
            "spec_mask": spec_mask,
            "a_final": a_final
        }


class PoissonSolver:
    """
    Solver for Poisson equations in gradient-domain processing.
    """
    
    def __init__(self):
        pass
    
    def generate_fused_gradient(self, ambient, flash, sigma=5.0, tau_s=0.12):
        """
        Generate fused gradient field from flash/no-flash pair.
        
        Args:
            ambient: Ambient (no-flash) image
            flash: Flash image
            sigma: Parameter for saturation weight calculation
            tau_s: Threshold for saturation weight calculation
            
        Returns:
            Dictionary containing gradient fields and weights
        """
        # Convert to grayscale for gradient calculation
        if ambient.shape[2] == 3:
            ambient_gray = cv2.cvtColor(ambient, cv2.COLOR_RGB2GRAY)
            flash_gray = cv2.cvtColor(flash, cv2.COLOR_RGB2GRAY)
        else:
            ambient_gray = ambient[:, :, 0]
            flash_gray = flash[:, :, 0]
        
        # Calculate gradients using Sobel operators
        ambient_grad_x = cv2.Sobel(ambient_gray, cv2.CV_64F, 1, 0, ksize=3)
        ambient_grad_y = cv2.Sobel(ambient_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        flash_grad_x = cv2.Sobel(flash_gray, cv2.CV_64F, 1, 0, ksize=3)
        flash_grad_y = cv2.Sobel(flash_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitudes
        ambient_grad_mag = np.sqrt(ambient_grad_x**2 + ambient_grad_y**2)
        flash_grad_mag = np.sqrt(flash_grad_x**2 + flash_grad_y**2)
        
        # Calculate weights based on gradient magnitudes
        # Use sigma parameter for weight calculation
        weights = 1.0 / (1.0 + (ambient_grad_mag / (sigma * flash_grad_mag + 1e-6))**2)
        
        # Apply smoothing to weights
        weights = gaussian_filter(weights, 2.0)
        
        # Calculate saturation weight based on flash intensity
        sat_weight = np.ones_like(flash_gray)
        sat_mask = flash_gray > tau_s
        sat_weight[sat_mask] = tau_s / flash_gray[sat_mask]
        
        # Smooth saturation weights
        sat_weight = gaussian_filter(sat_weight, 2.0)
        
        # Combine weights
        final_weights = weights * np.expand_dims(sat_weight, axis=-1) if len(weights.shape) > 2 else weights * sat_weight
        
        # Create fused gradients
        fused_grad_x = np.zeros_like(ambient)
        fused_grad_y = np.zeros_like(ambient)
        
        # For each color channel
        for c in range(ambient.shape[2]):
            # Calculate ambient and flash gradients for this channel
            a_grad_x = cv2.Sobel(ambient[:, :, c], cv2.CV_64F, 1, 0, ksize=3)
            a_grad_y = cv2.Sobel(ambient[:, :, c], cv2.CV_64F, 0, 1, ksize=3)
            
            f_grad_x = cv2.Sobel(flash[:, :, c], cv2.CV_64F, 1, 0, ksize=3)
            f_grad_y = cv2.Sobel(flash[:, :, c], cv2.CV_64F, 0, 1, ksize=3)
            
            # Fuse gradients using weights
            if len(final_weights.shape) > 2:
                fused_grad_x[:, :, c] = (1 - final_weights[:, :, c]) * a_grad_x + final_weights[:, :, c] * f_grad_x
                fused_grad_y[:, :, c] = (1 - final_weights[:, :, c]) * a_grad_y + final_weights[:, :, c] * f_grad_y
            else:
                fused_grad_x[:, :, c] = (1 - final_weights) * a_grad_x + final_weights * f_grad_x
                fused_grad_y[:, :, c] = (1 - final_weights) * a_grad_y + final_weights * f_grad_y
        
        return {
            "ambient_grad_x": ambient_grad_x,
            "ambient_grad_y": ambient_grad_y,
            "flash_grad_x": flash_grad_x,
            "flash_grad_y": flash_grad_y,
            "weights": weights,
            "sat_weight": sat_weight,
            "final_weights": final_weights,
            "fused_grad_x": fused_grad_x,
            "fused_grad_y": fused_grad_y
        }
    
    def solve_poisson(self, grad_x, grad_y, boundary_image, init_image=None, 
                      boundary_type="ambient", epsilon=1e-6, max_iterations=1000):
        """
        Solve Poisson equation to reconstruct an image from gradient fields.
        
        Args:
            grad_x: X gradient field
            grad_y: Y gradient field
            boundary_image: Image to use for boundary conditions
            init_image: Initial guess for the solution (default: None)
            boundary_type: Type of boundary conditions ("ambient", "flash", or "average")
            epsilon: Convergence parameter
            max_iterations: Maximum number of iterations
            
        Returns:
            Reconstructed image
        """
        height, width, channels = grad_x.shape
        
        # Initialize the solution
        if init_image is None:
            solution = np.zeros_like(boundary_image)
        else:
            solution = init_image.copy()
        
        # Set boundary conditions based on boundary_type
        if boundary_type == "ambient":
            boundary = boundary_image.copy()
        elif boundary_type == "flash":
            boundary = boundary_image.copy()
        elif boundary_type == "average":
            boundary = 0.5 * (boundary_image + boundary_image)
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        
        # Compute divergence of the gradient field
        div = np.zeros_like(grad_x)
        for c in range(channels):
            div[:, :, c] = cv2.Laplacian(solution[:, :, c], cv2.CV_64F) - (
                np.roll(grad_x[:, :, c], -1, axis=1) - grad_x[:, :, c] +
                np.roll(grad_y[:, :, c], -1, axis=0) - grad_y[:, :, c]
            )
        
        # Create mask for interior pixels (not on boundary)
        interior_mask = np.ones((height, width), dtype=bool)
        interior_mask[0, :] = False
        interior_mask[-1, :] = False
        interior_mask[:, 0] = False
        interior_mask[:, -1] = False
        
        # Iterative solution (Jacobi method)
        for iteration in range(max_iterations):
            prev_solution = solution.copy()
            
            for c in range(channels):
                # Update interior pixels
                solution[1:-1, 1:-1, c] = 0.25 * (
                    solution[0:-2, 1:-1, c] +  # Up
                    solution[2:, 1:-1, c] +    # Down
                    solution[1:-1, 0:-2, c] +  # Left
                    solution[1:-1, 2:, c] +    # Right
                    div[1:-1, 1:-1, c]         # Divergence
                )
            
            # Apply boundary conditions
            solution[0, :, :] = boundary[0, :, :]
            solution[-1, :, :] = boundary[-1, :, :]
            solution[:, 0, :] = boundary[:, 0, :]
            solution[:, -1, :] = boundary[:, -1, :]
            
            # Check convergence
            diff = np.max(np.abs(solution - prev_solution))
            if diff < epsilon:
                print(f"Converged after {iteration+1} iterations (diff: {diff})")
                break
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1}, diff: {diff}")
        else:
            print(f"Did not converge after {max_iterations} iterations (diff: {diff})")
        
        # Clip values to valid range
        solution = np.clip(solution, 0, 1)
        
        return solution


class GradientDomainProcessor:
    """
    Implementation of gradient-domain processing techniques for flash/no-flash photography.
    Based on Agrawal et al. "Removing Photography Artifacts using Gradient Projection and Flash-Exposure Sampling".
    """
    
    def __init__(self):
        self.poisson_solver = PoissonSolver()
    
    def process_flash_no_flash_pair(self, ambient, flash, sigma=5.0, tau_s=0.12,
                                    boundary_type="ambient", init_type="average",
                                    epsilon=1e-6, max_iterations=1000):
        """
        Process a flash/no-flash image pair using gradient-domain processing.
        
        Args:
            ambient: Ambient (no-flash) image
            flash: Flash image
            sigma: Parameter for weight calculation
            tau_s: Threshold for saturation weight calculation
            boundary_type: Type of boundary conditions
            init_type: Type of initialization for the solver
            epsilon: Convergence parameter
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing all intermediate and final results
        """
        # Store original images
        a_orig = ambient.copy()
        f_orig = flash.copy()
        
        # Step 1: Generate fused gradient field
        gradient_fields = self.poisson_solver.generate_fused_gradient(
            ambient, flash, sigma, tau_s
        )
        
        # Step 2: Initialize the solution based on init_type
        if init_type == "ambient":
            init_image = ambient.copy()
        elif init_type == "flash":
            init_image = flash.copy()
        elif init_type == "average":
            init_image = 0.5 * (ambient + flash)
        else:
            init_image = None
        
        # Step 3: Solve Poisson equation to reconstruct the image
        boundary_image = ambient if boundary_type == "ambient" else flash
        
        fused_image = self.poisson_solver.solve_poisson(
            gradient_fields["fused_grad_x"],
            gradient_fields["fused_grad_y"],
            boundary_image,
            init_image,
            boundary_type,
            epsilon,
            max_iterations
        )
        
        # Return results
        return {
            "a_orig": a_orig,
            "f_orig": f_orig,
            "gradient_fields": gradient_fields,
            "fused_image": fused_image
        }
    
    def plot_results(self, results, save_path=None):
        """
        Plot results of gradient-domain processing.
        
        Args:
            results: Dictionary containing gradient-domain processing results
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(15, 10))
        
        # Original images
        plt.subplot(2, 2, 1)
        plt.imshow(results["a_orig"])
        plt.title("Ambient (No Flash)")
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(results["f_orig"])
        plt.title("Flash")
        plt.axis('off')
        
        # Coherence map
        plt.subplot(2, 2, 3)
        weights = results["gradient_fields"]["weights"]
        if len(weights.shape) > 2:
            weights = np.mean(weights, axis=2)
        plt.imshow(weights, cmap='viridis')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title("Coherence Map (Weights)")
        plt.axis('off')
        
        # Fused result
        plt.subplot(2, 2, 4)
        plt.imshow(results["fused_image"])
        plt.title("Fused Result")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_gradient_fields(self, ambient, flash, gradient_fields, save_path=None):
        """
        Plot gradient fields for visualization.
        
        Args:
            ambient: Ambient (no-flash) image
            flash: Flash image
            gradient_fields: Dictionary containing gradient fields
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(15, 15))
        
        # Define how to visualize a gradient field
        def visualize_gradient(grad_x, grad_y, title, subplot_idx):
            plt.subplot(3, 3, subplot_idx)
            
            # Calculate magnitude and direction
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize for visualization
            if np.max(magnitude) > 0:
                magnitude = magnitude / np.max(magnitude)
            
            plt.imshow(magnitude, cmap='viridis')
            plt.title(title)
            plt.axis('off')
            
            return magnitude
        
        # Ambient image
        plt.subplot(3, 3, 1)
        plt.imshow(ambient)
        plt.title("Ambient (No Flash)")
        plt.axis('off')
        
        # Flash image
        plt.subplot(3, 3, 2)
        plt.imshow(flash)
        plt.title("Flash")
        plt.axis('off')
        
        # Weights
        plt.subplot(3, 3, 3)
        weights = gradient_fields["weights"]
        if len(weights.shape) > 2:
            weights = np.mean(weights, axis=2)
        plt.imshow(weights, cmap='viridis')
        plt.title("Weights")
        plt.axis('off')
        
        # Ambient gradient
        ambient_grad_mag = visualize_gradient(
            gradient_fields["ambient_grad_x"],
            gradient_fields["ambient_grad_y"],
            "Ambient Gradient Magnitude",
            4
        )
        
        # Flash gradient
        flash_grad_mag = visualize_gradient(
            gradient_fields["flash_grad_x"],
            gradient_fields["flash_grad_y"],
            "Flash Gradient Magnitude",
            5
        )
        
        # Saturation weight
        plt.subplot(3, 3, 6)
        plt.imshow(gradient_fields["sat_weight"], cmap='viridis')
        plt.title("Saturation Weight")
        plt.axis('off')
        
        # Fused gradient
        fused_grad_x_mean = np.mean(gradient_fields["fused_grad_x"], axis=2)
        fused_grad_y_mean = np.mean(gradient_fields["fused_grad_y"], axis=2)
        fused_grad_mag = visualize_gradient(
            fused_grad_x_mean,
            fused_grad_y_mean,
            "Fused Gradient Magnitude",
            7
        )
        
        # Difference magnitude
        plt.subplot(3, 3, 8)
        diff_mag = np.abs(ambient_grad_mag - flash_grad_mag)
        plt.imshow(diff_mag, cmap='viridis')
        plt.title("Gradient Difference Magnitude")
        plt.axis('off')
        
        # Final weights
        plt.subplot(3, 3, 9)
        final_weights = gradient_fields["final_weights"]
        if len(final_weights.shape) > 2:
            final_weights = np.mean(final_weights, axis=2)
        plt.imshow(final_weights, cmap='viridis')
        plt.title("Final Weights")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show() 