import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class BilateralFilter:
    def __init__(self):
        pass
    
    def basic_bilateral_filter(self, image, sigma_s, sigma_r):
        """
        Apply basic bilateral filter to an image
        
        Args:
            image: Input image (normalized to [0,1])
            sigma_s: Standard deviation for spatial Gaussian kernel
            sigma_r: Standard deviation for intensity Gaussian kernel
            
        Returns:
            Filtered image
        """
        # Make sure image is normalized
        if image.max() > 1.0:
            image = image / 255.0
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Initialize output image
        output = np.zeros_like(image)
        
        # Convert to float for processing
        image_float = image.astype(np.float32)
        
        # Determine kernel size (3 sigma rule)
        kernel_size = int(np.ceil(3 * sigma_s))
        
        # Apply bilateral filter
        if len(image.shape) == 3:  # Color image
            for c in range(image.shape[2]):
                output[:,:,c] = cv2.bilateralFilter(
                    image_float[:,:,c], 
                    d=kernel_size, 
                    sigmaColor=sigma_r, 
                    sigmaSpace=sigma_s
                )
        else:  # Grayscale image
            output = cv2.bilateralFilter(
                image_float, 
                d=kernel_size, 
                sigmaColor=sigma_r, 
                sigmaSpace=sigma_s
            )
            
        return output
    
    def joint_bilateral_filter(self, ambient_image, flash_image, sigma_s, sigma_r):
        """
        Apply joint bilateral filter to denoise ambient image using flash image
        
        Args:
            ambient_image: Image taken under ambient lighting (A)
            flash_image: Image taken with flash (F)
            sigma_s: Standard deviation for spatial Gaussian kernel
            sigma_r: Standard deviation for intensity Gaussian kernel
            
        Returns:
            Denoised ambient image (ANR)
        """
        # Make sure images are normalized
        if ambient_image.max() > 1.0:
            ambient_image = ambient_image / 255.0
        if flash_image.max() > 1.0:
            flash_image = flash_image / 255.0
            
        # Get image dimensions
        height, width = ambient_image.shape[:2]
        
        # Initialize output image
        output = np.zeros_like(ambient_image)
        
        # Determine kernel size (3 sigma rule)
        kernel_size = int(np.ceil(3 * sigma_s))
        
        # Apply joint bilateral filter using OpenCV's implementation
        if len(ambient_image.shape) == 3:  # Color image
            for c in range(ambient_image.shape[2]):
                # Convert flash image to grayscale for edge detection if it's the first channel
                if c == 0 and ambient_image.shape[2] == 3:
                    guide_image = cv2.cvtColor(flash_image, cv2.COLOR_BGR2GRAY)
                else:
                    guide_image = flash_image[:,:,c]
                    
                output[:,:,c] = cv2.ximgproc.jointBilateralFilter(
                    guide_image.astype(np.float32), 
                    ambient_image[:,:,c].astype(np.float32), 
                    d=kernel_size, 
                    sigmaColor=sigma_r, 
                    sigmaSpace=sigma_s
                )
        else:  # Grayscale image
            output = cv2.ximgproc.jointBilateralFilter(
                flash_image.astype(np.float32), 
                ambient_image.astype(np.float32), 
                d=kernel_size, 
                sigmaColor=sigma_r, 
                sigmaSpace=sigma_s
            )
            
        return output
    
    def detail_transfer(self, ambient_nr, flash_image, flash_base, epsilon=0.02):
        """
        Transfer details from flash image to ambient image
        
        Args:
            ambient_nr: Joint bilateral filtered ambient image (ANR)
            flash_image: Image taken with flash (F)
            flash_base: Bilateral filtered flash image (FBase)
            epsilon: Small value to avoid division by zero
            
        Returns:
            Detail enhanced ambient image (ADetail)
        """
        # Make sure images are normalized
        if ambient_nr.max() > 1.0:
            ambient_nr = ambient_nr / 255.0
        if flash_image.max() > 1.0:
            flash_image = flash_image / 255.0
        if flash_base.max() > 1.0:
            flash_base = flash_base / 255.0
            
        # Apply detail transfer equation: ADetail = ANR * (F + ε)/(FBase + ε)
        detail_factor = (flash_image + epsilon) / (flash_base + epsilon)
        detail_enhanced = ambient_nr * detail_factor
        
        # Clip values to valid range [0, 1]
        detail_enhanced = np.clip(detail_enhanced, 0.0, 1.0)
        
        return detail_enhanced
    
    def create_shadow_specularity_mask(self, flash_image, ambient_image, shadow_thresh=0.1, spec_thresh=0.9):
        """
        Create a mask to detect shadows and specularities
        
        Args:
            flash_image: Image taken with flash (F)
            ambient_image: Image taken under ambient lighting (A)
            shadow_thresh: Threshold for shadow detection
            spec_thresh: Threshold for specularity detection
            
        Returns:
            Binary mask (M) where 1 indicates shadow or specularity
        """
        # Make sure images are normalized
        if flash_image.max() > 1.0:
            flash_image = flash_image / 255.0
        if ambient_image.max() > 1.0:
            ambient_image = ambient_image / 255.0
            
        # Convert to grayscale if color images
        if len(flash_image.shape) == 3:
            flash_gray = cv2.cvtColor(flash_image, cv2.COLOR_BGR2GRAY)
        else:
            flash_gray = flash_image
            
        if len(ambient_image.shape) == 3:
            ambient_gray = cv2.cvtColor(ambient_image, cv2.COLOR_BGR2GRAY)
        else:
            ambient_gray = ambient_image
            
        # Create binary masks for shadows and specularities
        shadow_mask = flash_gray < shadow_thresh
        spec_mask = flash_gray > spec_thresh
        
        # Combine masks
        combined_mask = np.logical_or(shadow_mask, spec_mask).astype(np.float32)
        
        # Expand mask dimensions if needed
        if len(flash_image.shape) == 3:
            combined_mask = np.stack([combined_mask] * 3, axis=2)
            
        return combined_mask
    
    def final_image_fusion(self, a_detail, a_base, mask):
        """
        Fuse detail enhanced and base images using the shadow/specularity mask
        
        Args:
            a_detail: Detail enhanced ambient image (ADetail)
            a_base: Basic bilateral filtered ambient image (ABase)
            mask: Shadow and specularity mask (M)
            
        Returns:
            Final result image (AFinal)
        """
        # AFinal = (1 - M) * ADetail + M * ABase
        final_image = (1 - mask) * a_detail + mask * a_base
        
        # Clip values to valid range [0, 1]
        final_image = np.clip(final_image, 0.0, 1.0)
        
        return final_image
    
    def process_flash_no_flash_pair(self, ambient_image, flash_image, sigma_s_basic=8, sigma_r_basic=0.1, 
                                    sigma_s_joint=8, sigma_r_joint=0.1, epsilon=0.02, 
                                    shadow_thresh=0.1, spec_thresh=0.9):
        """
        Process a flash/no-flash image pair using the complete pipeline
        
        Args:
            ambient_image: Image taken under ambient lighting (A)
            flash_image: Image taken with flash (F)
            sigma_s_basic: Spatial sigma for basic bilateral filtering
            sigma_r_basic: Range sigma for basic bilateral filtering
            sigma_s_joint: Spatial sigma for joint bilateral filtering
            sigma_r_joint: Range sigma for joint bilateral filtering
            epsilon: Small value for detail transfer
            shadow_thresh: Threshold for shadow detection
            spec_thresh: Threshold for specularity detection
            
        Returns:
            Dictionary containing all intermediate and final results
        """
        # Step 1: Basic bilateral filtering of ambient image
        a_base = self.basic_bilateral_filter(ambient_image, sigma_s_basic, sigma_r_basic)
        
        # Step 2: Joint bilateral filtering
        a_nr = self.joint_bilateral_filter(ambient_image, flash_image, sigma_s_joint, sigma_r_joint)
        
        # Step 3: Basic bilateral filtering of flash image
        f_base = self.basic_bilateral_filter(flash_image, sigma_s_basic, sigma_r_basic)
        
        # Step 4: Detail transfer
        a_detail = self.detail_transfer(a_nr, flash_image, f_base, epsilon)
        
        # Step 5: Create shadow and specularity mask
        mask = self.create_shadow_specularity_mask(flash_image, ambient_image, shadow_thresh, spec_thresh)
        
        # Step 6: Final image fusion
        a_final = self.final_image_fusion(a_detail, a_base, mask)
        
        # Return all results for analysis
        return {
            "ambient": ambient_image,
            "flash": flash_image,
            "a_base": a_base,
            "a_nr": a_nr,
            "f_base": f_base,
            "a_detail": a_detail,
            "mask": mask,
            "a_final": a_final
        } 