import numpy as np
import cv2
from scipy import ndimage

class PoissonSolver:
    def __init__(self):
        """
        Initialize the Poisson solver
        """
        pass

    def laplacian_filtering(self, image):
        """
        Apply Laplacian filtering to an image
        
        Args:
            image: Input image
            
        Returns:
            Result of Laplacian filtering
        """
        # Define the Laplacian kernel
        laplacian_kernel = np.array([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype=np.float32)
        
        # Apply convolution
        if len(image.shape) == 3:  # Color image
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:,:,c] = cv2.filter2D(image[:,:,c], -1, laplacian_kernel)
        else:  # Grayscale image
            result = cv2.filter2D(image, -1, laplacian_kernel)
        
        return result

    def compute_gradient(self, image):
        """
        Compute the gradient of an image
        
        Args:
            image: Input image
            
        Returns:
            Tuple (dx, dy) containing the x and y derivatives
        """
        # Use Sobel operator to compute gradients
        if len(image.shape) == 3:  # Color image
            grad_x = np.zeros_like(image)
            grad_y = np.zeros_like(image)
            
            for c in range(image.shape[2]):
                grad_x[:,:,c] = cv2.Sobel(image[:,:,c], cv2.CV_32F, 1, 0, ksize=3)
                grad_y[:,:,c] = cv2.Sobel(image[:,:,c], cv2.CV_32F, 0, 1, ksize=3)
        else:  # Grayscale image
            grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        return grad_x, grad_y

    def compute_divergence(self, grad_x, grad_y):
        """
        Compute the divergence of a vector field
        
        Args:
            grad_x: X component of gradient field
            grad_y: Y component of gradient field
            
        Returns:
            Divergence of the vector field
        """
        # Compute partial derivatives of the gradient components
        if len(grad_x.shape) == 3:  # Color image
            div = np.zeros_like(grad_x)
            
            for c in range(grad_x.shape[2]):
                # Compute partial derivatives using Sobel
                grad_xx = cv2.Sobel(grad_x[:,:,c], cv2.CV_32F, 1, 0, ksize=3)
                grad_yy = cv2.Sobel(grad_y[:,:,c], cv2.CV_32F, 0, 1, ksize=3)
                
                # Divergence is the sum of partial derivatives
                div[:,:,c] = grad_xx + grad_yy
        else:  # Grayscale image
            grad_xx = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3)
            grad_yy = cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3)
            div = grad_xx + grad_yy
        
        return div

    def dot_product(self, a, b):
        """
        Compute the dot product of two vector fields
        
        Args:
            a: First vector field
            b: Second vector field
            
        Returns:
            Scalar dot product
        """
        # Make sure a and b have the same shape
        if a.shape != b.shape:
            if len(a.shape) > len(b.shape):
                # If a has more dimensions than b, expand b
                b = np.expand_dims(b, axis=-1)
                if len(a.shape) == 3:
                    b = np.repeat(b, a.shape[2], axis=2)
            elif len(b.shape) > len(a.shape):
                # If b has more dimensions than a, expand a
                a = np.expand_dims(a, axis=-1)
                if len(b.shape) == 3:
                    a = np.repeat(a, b.shape[2], axis=2)
        
        return np.sum(a * b)

    def conjugate_gradient_descent(self, divergence, boundary_mask, boundary_values, 
                                  init=None, epsilon=1e-6, max_iterations=1000):
        """
        Solve the Poisson equation using conjugate gradient descent
        
        Args:
            divergence: Divergence of the gradient field (D)
            boundary_mask: Binary mask (B) where 0 indicates boundary, 1 elsewhere
            boundary_values: Values to use at the boundary (I_boundary)
            init: Initial guess for the solution (I_init), if None, use zeros
            epsilon: Convergence parameter (ε)
            max_iterations: Maximum number of iterations (N)
            
        Returns:
            Integrated image (I*)
        """
        # Get dimensions
        height, width = divergence.shape[:2]
        
        # Initialize solution with zeros if not provided
        if init is None:
            I_star = np.zeros_like(divergence, dtype=np.float32)
        else:
            I_star = init.copy().astype(np.float32)
        
        # Apply boundary conditions
        I_star = boundary_mask * I_star + (1 - boundary_mask) * boundary_values
        
        # Initialize residual
        laplacian_I_star = self.laplacian_filtering(I_star)
        
        # Make sure divergence has the same shape as laplacian_I_star for proper broadcasting
        if len(divergence.shape) != len(laplacian_I_star.shape):
            if len(divergence.shape) == 2 and len(laplacian_I_star.shape) == 3:
                # Expand divergence to match the channels in laplacian_I_star
                divergence = np.expand_dims(divergence, axis=-1)
                divergence = np.repeat(divergence, laplacian_I_star.shape[2], axis=2)
            elif len(divergence.shape) == 3 and len(laplacian_I_star.shape) == 2:
                # Take average of divergence channels
                divergence = np.mean(divergence, axis=2)
        
        r = boundary_mask * (divergence - laplacian_I_star)
        
        # Initialize search direction
        d = r.copy()
        
        # Compute initial delta
        delta_new = self.dot_product(r, r)
        
        # Check if already converged
        if np.sqrt(delta_new) < epsilon:
            return I_star
        
        # Main loop
        iteration = 0
        while np.sqrt(delta_new) > epsilon and iteration < max_iterations:
            # Compute q
            q = self.laplacian_filtering(d)
            
            # Compute step size
            alpha = delta_new / self.dot_product(d, q)
            
            # Update solution
            I_star = I_star + boundary_mask * (alpha * d)
            
            # Update residual
            if (iteration + 1) % 50 == 0:  # Restart residual every 50 iterations to avoid numerical drift
                r = boundary_mask * (divergence - self.laplacian_filtering(I_star))
            else:
                r = r - boundary_mask * (alpha * q)
            
            # Compute new delta
            delta_old = delta_new
            delta_new = self.dot_product(r, r)
            
            # Compute beta
            beta = delta_new / delta_old
            
            # Update search direction
            d = r + beta * d
            
            # Increment iteration counter
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Residual: {np.sqrt(delta_new)}")
        
        print(f"Poisson solver converged after {iteration} iterations, Final residual: {np.sqrt(delta_new)}")
        
        return I_star
    
    def differentiate_and_reintegrate(self, image, epsilon=1e-6, max_iterations=1000):
        """
        Differentiate an image and then reintegrate it using the Poisson solver
        
        Args:
            image: Input image
            epsilon: Convergence parameter for CGD
            max_iterations: Maximum number of iterations for CGD
            
        Returns:
            Reintegrated image
        """
        # Compute gradients
        grad_x, grad_y = self.compute_gradient(image)
        
        # Compute divergence
        divergence = self.compute_divergence(grad_x, grad_y)
        
        # Alternative approach: Apply Laplacian directly to the image
        # divergence = self.laplacian_filtering(image)
        
        # Create boundary mask and values
        boundary_mask = np.ones_like(image)
        if len(image.shape) == 3:  # Color image
            boundary_mask[0, :, :] = 0  # Top edge
            boundary_mask[-1, :, :] = 0  # Bottom edge
            boundary_mask[:, 0, :] = 0  # Left edge
            boundary_mask[:, -1, :] = 0  # Right edge
        else:  # Grayscale image
            boundary_mask[0, :] = 0  # Top edge
            boundary_mask[-1, :] = 0  # Bottom edge
            boundary_mask[:, 0] = 0  # Left edge
            boundary_mask[:, -1] = 0  # Right edge
        
        # Use original image values at the boundary
        boundary_values = image.copy()
        
        # Initial guess (zeros)
        init = np.zeros_like(image)
        
        # Solve the Poisson equation
        print("Solving Poisson equation...")
        reintegrated = self.conjugate_gradient_descent(
            divergence, boundary_mask, boundary_values, init, epsilon, max_iterations
        )
        
        return reintegrated
    
    def generate_fused_gradient(self, ambient_image, flash_image, sigma=5.0, tau_s=0.12):
        """
        Generate a fused gradient field from ambient and flash images
        
        Args:
            ambient_image: Ambient image (a)
            flash_image: Flash image (Φ')
            sigma: Parameter for saturation weight calculation
            tau_s: Threshold for saturation weight calculation
            
        Returns:
            Dictionary containing fused gradient field and intermediate results
        """
        # Ensure images are float and normalized
        if ambient_image.max() > 1.0:
            ambient_image = ambient_image.astype(np.float32) / 255.0
        if flash_image.max() > 1.0:
            flash_image = flash_image.astype(np.float32) / 255.0
        
        # Compute gradients of ambient and flash images
        grad_a_x, grad_a_y = self.compute_gradient(ambient_image)
        grad_f_x, grad_f_y = self.compute_gradient(flash_image)
        
        # Step 1: Compute the gradient orientation coherency map M
        # Numerator: |∇Φ' · ∇a| = |Φ'_x · a_x + Φ'_y · a_y|
        dot_product = np.abs(grad_f_x * grad_a_x + grad_f_y * grad_a_y)
        
        # Denominator: ||∇Φ'|| ||∇a|| = sqrt(Φ'_x^2 + Φ'_y^2) * sqrt(a_x^2 + a_y^2)
        norm_grad_f = np.sqrt(grad_f_x**2 + grad_f_y**2 + 1e-10)  # Add small constant to avoid division by zero
        norm_grad_a = np.sqrt(grad_a_x**2 + grad_a_y**2 + 1e-10)
        
        # Coherency map
        coherency_map = dot_product / (norm_grad_f * norm_grad_a)
        
        # Ensure coherency_map is in [0, 1]
        coherency_map = np.clip(coherency_map, 0.0, 1.0)
        
        # Step 2: Compute the pixel-wise saturation weight map w_s
        saturation_weight = np.tanh(sigma * (flash_image - tau_s))
        
        # Normalize saturation_weight to [0, 1]
        saturation_weight = (saturation_weight - np.min(saturation_weight)) / (np.max(saturation_weight) - np.min(saturation_weight) + 1e-10)
        
        # Step 3: Compute the fused gradient field
        # ∇Φ* = w_s · ∇a + (1 - w_s)(M · ∇Φ' + (1 - M) · ∇a)
        
        # For x component
        fused_grad_x = (
            saturation_weight * grad_a_x + 
            (1 - saturation_weight) * (
                coherency_map * grad_f_x + 
                (1 - coherency_map) * grad_a_x
            )
        )
        
        # For y component
        fused_grad_y = (
            saturation_weight * grad_a_y + 
            (1 - saturation_weight) * (
                coherency_map * grad_f_y + 
                (1 - coherency_map) * grad_a_y
            )
        )
        
        return {
            "grad_a_x": grad_a_x,
            "grad_a_y": grad_a_y,
            "grad_f_x": grad_f_x,
            "grad_f_y": grad_f_y,
            "coherency_map": coherency_map,
            "saturation_weight": saturation_weight,
            "fused_grad_x": fused_grad_x,
            "fused_grad_y": fused_grad_y
        }
    
    def integrate_fused_gradient(self, fused_grad_x, fused_grad_y, ambient_image, flash_image,
                                boundary_type="ambient", init_type="average", 
                                epsilon=1e-6, max_iterations=1000):
        """
        Integrate the fused gradient field to create the final image
        
        Args:
            fused_grad_x: X component of fused gradient field
            fused_grad_y: Y component of fused gradient field
            ambient_image: Ambient image
            flash_image: Flash image
            boundary_type: Type of boundary conditions ("ambient", "flash", or "average")
            init_type: Type of initialization ("ambient", "flash", "average", or "zero")
            epsilon: Convergence parameter for CGD
            max_iterations: Maximum number of iterations for CGD
            
        Returns:
            Integrated fused image
        """
        # Compute divergence of the fused gradient field
        print("Integrating fused gradient field...")
        divergence = self.compute_divergence(fused_grad_x, fused_grad_y)
        
        # Create boundary mask
        boundary_mask = np.ones_like(ambient_image)
        boundary_mask[0, :] = 0  # Top edge
        boundary_mask[-1, :] = 0  # Bottom edge
        boundary_mask[:, 0] = 0  # Left edge
        boundary_mask[:, -1] = 0  # Right edge
        
        # Set boundary values based on boundary_type
        if boundary_type == "ambient":
            boundary_values = ambient_image.copy()
        elif boundary_type == "flash":
            boundary_values = flash_image.copy()
        elif boundary_type == "average":
            boundary_values = (ambient_image + flash_image) / 2
        else:
            raise ValueError("Invalid boundary_type. Must be 'ambient', 'flash', or 'average'.")
        
        # Set initial guess based on init_type
        if init_type == "ambient":
            init = ambient_image.copy()
        elif init_type == "flash":
            init = flash_image.copy()
        elif init_type == "average":
            init = (ambient_image + flash_image) / 2
        elif init_type == "zero":
            init = np.zeros_like(ambient_image)
        else:
            raise ValueError("Invalid init_type. Must be 'ambient', 'flash', 'average', or 'zero'.")
        
        # Solve the Poisson equation
        print(f"Integrating fused gradient field with {boundary_type} boundary and {init_type} initialization...")
        fused_image = self.conjugate_gradient_descent(
            divergence, boundary_mask, boundary_values, init, epsilon, max_iterations
        )
        
        return fused_image 