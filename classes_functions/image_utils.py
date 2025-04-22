import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

class ImageUtils:
    def __init__(self):
        pass
    
    def load_image(self, path, normalize=True):
        """
        Load an image from disk
        
        Args:
            path: Path to the image file
            normalize: Whether to normalize image to [0,1] range
            
        Returns:
            Loaded image
        """
        # Read image
        img = cv2.imread(path)
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize if requested
        if normalize and img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
            
        return img
    
    def save_image(self, img, path, denormalize=True):
        """
        Save an image to disk
        
        Args:
            img: Image to save
            path: Path where to save the image
            denormalize: Whether to denormalize from [0,1] to [0,255]
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Make a copy to avoid modifying the original
        img_save = img.copy()
        
        # Denormalize if requested
        if denormalize and img_save.max() <= 1.0:
            img_save = (img_save * 255).astype(np.uint8)
        
        # Convert from RGB to BGR
        img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(path, img_save)
    
    def load_flash_no_flash_pair(self, dataset_dir, image_name, normalize=True):
        """
        Load a flash/no-flash image pair
        
        Args:
            dataset_dir: Directory containing flash and nonflash subdirectories
            image_name: Name of the image file
            normalize: Whether to normalize images to [0,1] range
            
        Returns:
            Tuple of (ambient_image, flash_image)
        """
        # Construct paths
        ambient_path = os.path.join(dataset_dir, 'nonflash', image_name)
        flash_path = os.path.join(dataset_dir, 'flash', image_name)
        
        # Load images
        ambient_image = self.load_image(ambient_path, normalize)
        flash_image = self.load_image(flash_path, normalize)
        
        return ambient_image, flash_image
    
    def calculate_difference(self, img1, img2):
        """
        Calculate difference between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Difference image (absolute difference)
        """
        return np.abs(img1 - img2)
    
    def calculate_psnr(self, img1, img2):
        """
        Calculate Peak Signal-to-Noise Ratio between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            PSNR value
        """
        # Make sure images are normalized
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0
            
        # Calculate MSE
        mse = np.mean((img1 - img2) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
            
        # Calculate PSNR
        psnr = 10 * np.log10(1.0 / mse)
        
        return psnr
    
    def plot_images(self, images, titles=None, figsize=(15, 10), rows=None, cols=None, cmap=None, save_path=None):
        """
        Plot multiple images in a grid
        
        Args:
            images: List of images to plot
            titles: List of titles for each image
            figsize: Figure size (width, height)
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            cmap: Colormap for grayscale images
            save_path: Path to save the figure (if None, figure is displayed)
        """
        # Determine number of rows and columns
        n = len(images)
        if rows is None and cols is None:
            cols = 3
            rows = (n + cols - 1) // cols
        elif rows is None:
            rows = (n + cols - 1) // cols
        elif cols is None:
            cols = (n + rows - 1) // rows
            
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes if needed
        if rows * cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
            
        # Plot each image
        for i, img in enumerate(images):
            if i < len(axes):
                # Handle single-channel images
                if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                    axes[i].imshow(img, cmap=cmap or 'gray')
                else:
                    # Make sure image is in [0,1] range for display
                    if img.max() > 1.0:
                        img_display = img / 255.0
                    else:
                        img_display = img
                    axes[i].imshow(img_display)
                    
                # Set title if provided
                if titles and i < len(titles):
                    axes[i].set_title(titles[i])
                    
                # Remove axis ticks
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                
        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')
            
        # Adjust layout
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison(self, results_dict, save_dir=None):
        """
        Plot comparison of all processing steps for flash/no-flash photography
        
        Args:
            results_dict: Dictionary with results from process_flash_no_flash_pair
            save_dir: Directory to save comparison plots (if None, plots are displayed)
        """
        # Create main comparison figure
        main_images = [
            results_dict["ambient"],
            results_dict["flash"],
            results_dict["a_base"],
            results_dict["a_nr"],
            results_dict["a_detail"],
            results_dict["a_final"]
        ]
        
        main_titles = [
            "Ambient Image",
            "Flash Image",
            "Basic Bilateral Filter",
            "Joint Bilateral Filter",
            "Detail Transfer",
            "Final Result"
        ]
        
        if save_dir:
            save_path = os.path.join(save_dir, "comparison.png")
        else:
            save_path = None
            
        self.plot_images(main_images, main_titles, figsize=(18, 10), rows=2, cols=3, save_path=save_path)
        
        # Create difference images
        diff_a_base = self.calculate_difference(results_dict["ambient"], results_dict["a_base"])
        diff_a_nr = self.calculate_difference(results_dict["ambient"], results_dict["a_nr"])
        diff_a_detail = self.calculate_difference(results_dict["a_nr"], results_dict["a_detail"])
        diff_final = self.calculate_difference(results_dict["a_detail"], results_dict["a_final"])
        
        diff_images = [diff_a_base, diff_a_nr, diff_a_detail, diff_final, results_dict["mask"]]
        diff_titles = [
            "Ambient vs Basic Bilateral",
            "Ambient vs Joint Bilateral",
            "Joint Bilateral vs Detail Transfer",
            "Detail Transfer vs Final",
            "Shadow/Specularity Mask"
        ]
        
        if save_dir:
            save_path = os.path.join(save_dir, "differences.png")
        else:
            save_path = None
            
        self.plot_images(diff_images, diff_titles, figsize=(18, 6), rows=1, cols=5, save_path=save_path)
        
        # Save individual images if save_dir is provided
        if save_dir:
            for name, img in results_dict.items():
                self.save_image(img, os.path.join(save_dir, f"{name}.png"))
                
            # Save difference images
            self.save_image(diff_a_base, os.path.join(save_dir, "diff_a_base.png"))
            self.save_image(diff_a_nr, os.path.join(save_dir, "diff_a_nr.png"))
            self.save_image(diff_a_detail, os.path.join(save_dir, "diff_a_detail.png"))
            self.save_image(diff_final, os.path.join(save_dir, "diff_final.png"))
    
    def batch_process_dataset(self, bilateral_filter, dataset_dir, output_dir, image_names=None, 
                              sigma_s_basic=8, sigma_r_basic=0.1, sigma_s_joint=8, 
                              sigma_r_joint=0.1, epsilon=0.02):
        """
        Process multiple flash/no-flash image pairs
        
        Args:
            bilateral_filter: BilateralFilter instance
            dataset_dir: Directory containing flash and nonflash subdirectories
            output_dir: Directory to save results
            image_names: List of image names to process (if None, all images in nonflash dir are processed)
            sigma_s_basic: Spatial sigma for basic bilateral filtering
            sigma_r_basic: Range sigma for basic bilateral filtering
            sigma_s_joint: Spatial sigma for joint bilateral filtering
            sigma_r_joint: Range sigma for joint bilateral filtering
            epsilon: Small value for detail transfer
            
        Returns:
            Dictionary mapping image names to result dictionaries
        """
        # If no image names provided, find all images in nonflash directory
        if image_names is None:
            nonflash_dir = os.path.join(dataset_dir, 'nonflash')
            image_names = [f for f in os.listdir(nonflash_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
        # Process each image pair
        results = {}
        for img_name in image_names:
            print(f"Processing {img_name}...")
            
            # Load image pair
            ambient_image, flash_image = self.load_flash_no_flash_pair(dataset_dir, img_name)
            
            # Process images
            result = bilateral_filter.process_flash_no_flash_pair(
                ambient_image, flash_image,
                sigma_s_basic=sigma_s_basic, 
                sigma_r_basic=sigma_r_basic,
                sigma_s_joint=sigma_s_joint, 
                sigma_r_joint=sigma_r_joint,
                epsilon=epsilon
            )
            
            # Save results
            img_output_dir = os.path.join(output_dir, os.path.splitext(img_name)[0])
            os.makedirs(img_output_dir, exist_ok=True)
            
            # Plot comparison
            self.plot_comparison(result, save_dir=img_output_dir)
            
            # Store results
            results[img_name] = result
            
        return results
    
    def parameter_sweep(self, bilateral_filter, ambient_image, flash_image, param_name, 
                        param_values, fixed_params=None, output_dir=None):
        """
        Perform parameter sweep and compare results
        
        Args:
            bilateral_filter: BilateralFilter instance
            ambient_image: Ambient image
            flash_image: Flash image
            param_name: Name of parameter to sweep
            param_values: List of values for the parameter
            fixed_params: Dictionary of fixed parameter values
            output_dir: Directory to save results
            
        Returns:
            List of result dictionaries for each parameter value
        """
        # Set default fixed parameters
        if fixed_params is None:
            fixed_params = {
                'sigma_s_basic': 8, 
                'sigma_r_basic': 0.1,
                'sigma_s_joint': 8, 
                'sigma_r_joint': 0.1,
                'epsilon': 0.02,
                'shadow_thresh': 0.1,
                'spec_thresh': 0.9
            }
            
        # Process image pair with each parameter value
        results = []
        final_images = []
        param_strs = []
        
        for value in param_values:
            # Set parameter value
            params = fixed_params.copy()
            params[param_name] = value
            
            # Process image pair
            result = bilateral_filter.process_flash_no_flash_pair(
                ambient_image, flash_image, **params
            )
            
            # Store results
            results.append(result)
            final_images.append(result['a_final'])
            param_strs.append(f"{param_name}={value}")
            
            # Save results if output_dir is provided
            if output_dir:
                img_output_dir = os.path.join(output_dir, f"{param_name}_{value}")
                os.makedirs(img_output_dir, exist_ok=True)
                self.plot_comparison(result, save_dir=img_output_dir)
        
        # Plot comparison of final results
        if output_dir:
            save_path = os.path.join(output_dir, f"{param_name}_sweep.png")
        else:
            save_path = None
            
        self.plot_images(final_images, param_strs, figsize=(18, 10), save_path=save_path)
        
        return results 