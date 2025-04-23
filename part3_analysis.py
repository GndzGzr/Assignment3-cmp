#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash/No-Flash Photography - Part 3 Analysis and Presentation

This script creates an HTML presentation with the results of flash/no-flash processing,
comparing bilateral filtering and gradient domain approaches.

Usage:
    python part3_analysis.py --results_dir part3_results

This will generate an HTML presentation for easy viewing of the results.
"""

import os
import argparse
import glob
import base64
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flash/No-Flash Photography Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
        }
        .image-item {
            text-align: center;
            max-width: 100%;
        }
        .image-item img {
            max-width: 100%;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        .image-caption {
            margin-top: 8px;
            font-size: 0.9em;
            color: #555;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .parameter-table {
            width: 100%;
            margin: 10px 0;
        }
        .comparison-row {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .comparison-cell {
            flex: 1;
            min-width: 300px;
            padding: 10px;
        }
        .highlight {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .conclusion {
            font-style: italic;
            background-color: #f0f4c3;
            padding: 15px;
            border-radius: 5px;
        }
        .navigation {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }
        .nav-button {
            text-decoration: none;
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .nav-button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>Flash/No-Flash Photography Analysis</h1>
    <p>This report presents the analysis of various techniques for processing flash/no-flash image pairs.</p>
    
    <!-- Introduction -->
    <div class="section">
        <h2>Introduction</h2>
        <p>Flash/no-flash photography techniques aim to combine the best aspects of both images:</p>
        <ul>
            <li>The natural lighting and colors from the ambient (no-flash) image</li>
            <li>The low noise and sharp details from the flash image</li>
        </ul>
        <p>This analysis compares two main approaches:</p>
        <ol>
            <li><strong>Bilateral Filtering</strong>: Based on Petschnigg et al. "Digital Photography with Flash and No-Flash Image Pairs"</li>
            <li><strong>Gradient Domain Processing</strong>: Based on Agrawal et al. "Removing Photography Artifacts using Gradient Projection and Flash-Exposure Sampling"</li>
        </ol>
    </div>

    <!-- Bilateral Filtering Results -->
    <div class="section">
        <h2>Bilateral Filtering Results</h2>
        <p>This technique uses edge-preserving bilateral filters to transfer details from the flash image while maintaining the ambient lighting.</p>
        
        {bilateral_results_html}
    </div>

    <!-- Gradient Domain Results -->
    <div class="section">
        <h2>Gradient Domain Processing Results</h2>
        <p>This technique operates in the gradient domain, creating a fused gradient field from both images and then solving a Poisson equation to reconstruct the final image.</p>
        
        {gradient_results_html}
    </div>

    <!-- Parameter Analysis -->
    <div class="section">
        <h2>Parameter Analysis</h2>
        
        <h3>Bilateral Filtering Parameters</h3>
        <table class="parameter-table">
            <tr>
                <th>Parameter</th>
                <th>Description</th>
                <th>Typical Values</th>
                <th>Effect</th>
            </tr>
            <tr>
                <td>Spatial Sigma (sigma_s)</td>
                <td>Controls the spatial extent of the filter</td>
                <td>8-32 pixels</td>
                <td>Larger values blur over larger regions</td>
            </tr>
            <tr>
                <td>Range Sigma (sigma_r)</td>
                <td>Controls edge preservation</td>
                <td>0.05-0.2</td>
                <td>Smaller values preserve stronger edges</td>
            </tr>
            <tr>
                <td>Detail Strength (epsilon)</td>
                <td>Controls amount of detail transfer</td>
                <td>0.01-0.05</td>
                <td>Higher values increase detail but may introduce noise</td>
            </tr>
        </table>
        
        {bilateral_parameter_html}
        
        <h3>Gradient Domain Parameters</h3>
        <table class="parameter-table">
            <tr>
                <th>Parameter</th>
                <th>Description</th>
                <th>Typical Values</th>
                <th>Effect</th>
            </tr>
            <tr>
                <td>Sigma</td>
                <td>Controls weight calculation for gradient fusion</td>
                <td>1-10</td>
                <td>Higher values increase flash influence on gradients</td>
            </tr>
            <tr>
                <td>Tau_s</td>
                <td>Threshold for saturation weight calculation</td>
                <td>0.05-0.2</td>
                <td>Controls how the algorithm handles bright areas</td>
            </tr>
            <tr>
                <td>Boundary Conditions</td>
                <td>Define values at image boundary for Poisson solving</td>
                <td>ambient, flash, average</td>
                <td>Affect color tone of final result</td>
            </tr>
        </table>
        
        {gradient_parameter_html}
    </div>

    <!-- Technique Comparison -->
    <div class="section">
        <h2>Technique Comparison</h2>
        
        <div class="comparison-row">
            <div class="comparison-cell">
                <h3>Bilateral Filtering</h3>
                <h4>Advantages</h4>
                <ul>
                    <li>Faster computation time</li>
                    <li>Effective noise reduction while preserving edges</li>
                    <li>Simpler implementation with fewer parameters</li>
                    <li>Good for detail enhancement from flash image</li>
                </ul>
                
                <h4>Disadvantages</h4>
                <ul>
                    <li>Can produce halo artifacts around strong edges</li>
                    <li>Less effective for shadow and specular handling</li>
                    <li>May require more parameter tuning for optimal results</li>
                </ul>
                
                <h4>Best Use Cases</h4>
                <ul>
                    <li>Denoising low-light images</li>
                    <li>Indoor photography in dimly lit environments</li>
                    <li>When computational resources are limited</li>
                </ul>
            </div>
            
            <div class="comparison-cell">
                <h3>Gradient Domain Processing</h3>
                <h4>Advantages</h4>
                <ul>
                    <li>Better preservation of edge transitions</li>
                    <li>More natural handling of shadows and highlights</li>
                    <li>Less prone to halo artifacts</li>
                    <li>Better for complex scenes with mixed lighting</li>
                </ul>
                
                <h4>Disadvantages</h4>
                <ul>
                    <li>More computationally intensive</li>
                    <li>Requires solving Poisson equation (iterative process)</li>
                    <li>More complex implementation</li>
                    <li>Results depend on proper boundary conditions</li>
                </ul>
                
                <h4>Best Use Cases</h4>
                <ul>
                    <li>Scenes with strong specular highlights</li>
                    <li>Complex lighting with shadows cast by flash</li>
                    <li>When highest quality results are needed</li>
                </ul>
            </div>
        </div>
        
        {comparison_html}
    </div>

    <!-- Guidelines -->
    <div class="section">
        <h2>Guidelines for Capturing Good Flash/No-Flash Pairs</h2>
        
        <div class="highlight">
            <h3>Camera Setup</h3>
            <ul>
                <li>Use a stable tripod to ensure both images are perfectly aligned</li>
                <li>Use manual focus to keep focus consistent between shots</li>
                <li>Use manual exposure settings for the no-flash (ambient) image</li>
                <li>Set a fixed white balance (not auto)</li>
            </ul>
            
            <h3>Flash Control</h3>
            <ul>
                <li>Use an external flash if possible (more control over flash power)</li>
                <li>The flash image should be properly exposed (not overexposed)</li>
                <li>For the no-flash image, use a longer exposure time</li>
            </ul>
            
            <h3>Subject Selection</h3>
            <ul>
                <li>For bilateral filtering: dimly lit environments with details</li>
                <li>For gradient domain: scenes with mixed specular and matte surfaces</li>
                <li>Avoid scenes with moving objects</li>
            </ul>
            
            <h3>Common Issues to Avoid</h3>
            <ul>
                <li>Camera movement between shots</li>
                <li>Subject movement between shots</li>
                <li>Flash shadows or harsh specular highlights</li>
                <li>Flash image overexposure</li>
                <li>No-flash image too dark or noisy</li>
            </ul>
        </div>
    </div>

    <!-- Conclusion -->
    <div class="section">
        <h2>Conclusion</h2>
        
        <p class="conclusion">
            Both bilateral filtering and gradient domain processing offer effective solutions for combining flash/no-flash image pairs, with different strengths and weaknesses.
            The choice between them depends on the specific characteristics of the scene and the desired quality-speed tradeoff.
        </p>
        
        <p>For optimal results:</p>
        <ol>
            <li>Start with bilateral filtering for its speed and simplicity</li>
            <li>If results show haloing or poor shadow handling, try gradient domain processing</li>
            <li>Experiment with parameters to find the best settings for each specific image pair</li>
        </ol>
    </div>

    <div class="navigation">
        <a href="#" class="nav-button">Back to Top</a>
    </div>
</body>
</html>
"""

def encode_image_to_base64(img_path):
    """Convert an image file to base64 for embedding in HTML."""
    if not os.path.exists(img_path):
        return None
    
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    extension = os.path.splitext(img_path)[1].lower()
    if extension == '.jpg' or extension == '.jpeg':
        mime_type = 'image/jpeg'
    elif extension == '.png':
        mime_type = 'image/png'
    else:
        mime_type = 'image/jpeg'  # Default
    
    return f"data:{mime_type};base64,{encoded_string}"

def create_image_html(img_path, caption, max_width=None):
    """Create HTML for an image with caption."""
    base64_image = encode_image_to_base64(img_path)
    
    if not base64_image:
        return f"<p>Image not found: {os.path.basename(img_path)}</p>"
    
    style = f'style="max-width: {max_width}px;"' if max_width else ""
    
    return f"""
    <div class="image-item">
        <img src="{base64_image}" alt="{caption}" {style}>
        <div class="image-caption">{caption}</div>
    </div>
    """

def create_bilateral_results_html(results_dir):
    """Create HTML for bilateral filtering results."""
    bilateral_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_bilateral")) 
                     if os.path.isdir(d)]
    
    if not bilateral_dirs:
        return "<p>No bilateral filtering results found.</p>"
    
    html = ""
    
    for bilateral_dir in bilateral_dirs:
        name = os.path.basename(bilateral_dir).replace("_bilateral", "")
        
        html += f"<h3>Sample: {name}</h3>"
        html += "<div class='image-container'>"
        
        # Original images
        ambient_path = os.path.join(results_dir, "..", "sample_images", "bilateral", name, "noflash.jpg")
        flash_path = os.path.join(results_dir, "..", "sample_images", "bilateral", name, "flash.jpg")
        
        if os.path.exists(ambient_path) and os.path.exists(flash_path):
            html += create_image_html(ambient_path, "Ambient (No Flash)")
            html += create_image_html(flash_path, "Flash")
        
        # Result images
        result_path = os.path.join(bilateral_dir, "bilateral_final.jpg")
        if os.path.exists(result_path):
            html += create_image_html(result_path, "Bilateral Filtered Result")
        
        html += "</div>"
        
        # Comparison image
        comparison_path = os.path.join(bilateral_dir, "bilateral_results.png")
        if os.path.exists(comparison_path):
            html += "<h4>Detailed Processing Steps</h4>"
            html += "<div class='image-container'>"
            html += create_image_html(comparison_path, "Bilateral Filtering Process Steps", 800)
            html += "</div>"
    
    return html

def create_gradient_results_html(results_dir):
    """Create HTML for gradient domain results."""
    gradient_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_gradient")) 
                    if os.path.isdir(d)]
    
    if not gradient_dirs:
        return "<p>No gradient domain results found.</p>"
    
    html = ""
    
    for gradient_dir in gradient_dirs:
        name = os.path.basename(gradient_dir).replace("_gradient", "")
        
        html += f"<h3>Sample: {name}</h3>"
        html += "<div class='image-container'>"
        
        # Original images
        ambient_path = os.path.join(results_dir, "..", "sample_images", "gradient", name, "noflash.jpg")
        flash_path = os.path.join(results_dir, "..", "sample_images", "gradient", name, "flash.jpg")
        
        if os.path.exists(ambient_path) and os.path.exists(flash_path):
            html += create_image_html(ambient_path, "Ambient (No Flash)")
            html += create_image_html(flash_path, "Flash")
        
        # Result images
        result_path = os.path.join(gradient_dir, "gradient_final.jpg")
        if os.path.exists(result_path):
            html += create_image_html(result_path, "Gradient Domain Result")
        
        html += "</div>"
        
        # Gradient fields
        fields_path = os.path.join(gradient_dir, "gradient_fields.png")
        if os.path.exists(fields_path):
            html += "<h4>Gradient Fields</h4>"
            html += "<div class='image-container'>"
            html += create_image_html(fields_path, "Gradient Fields and Weights", 800)
            html += "</div>"
        
        # Results visualization
        results_path = os.path.join(gradient_dir, "gradient_results.png")
        if os.path.exists(results_path):
            html += "<h4>Results Summary</h4>"
            html += "<div class='image-container'>"
            html += create_image_html(results_path, "Gradient Domain Results Summary", 800)
            html += "</div>"
    
    return html

def create_parameter_analysis_html(results_dir, technique):
    """Create HTML for parameter analysis."""
    if technique == "bilateral":
        param_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_bilateral", "parameter_analysis")) 
                     if os.path.isdir(d)]
        param_prefixes = ["bilateral_sigma_s", "bilateral_sigma_r", "bilateral_epsilon"]
        param_names = ["Spatial Sigma", "Range Sigma", "Detail Strength"]
    else:  # gradient
        param_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_gradient", "parameter_analysis")) 
                     if os.path.isdir(d)]
        param_prefixes = ["gradient_sigma", "gradient_tau_s", "gradient_boundary"]
        param_names = ["Sigma", "Tau_s", "Boundary Conditions"]
    
    if not param_dirs:
        return f"<p>No parameter analysis found for {technique} technique.</p>"
    
    html = ""
    
    for param_dir in param_dirs[:1]:  # Take just the first one to avoid repetition
        for prefix, name in zip(param_prefixes, param_names):
            param_files = glob.glob(os.path.join(param_dir, f"{prefix}*.png"))
            
            if param_files:
                html += f"<h4>Effect of {name}</h4>"
                html += "<div class='image-container'>"
                
                for param_file in param_files:
                    html += create_image_html(param_file, f"{name} Comparison", 800)
                
                html += "</div>"
    
    return html

def create_technique_comparison_html(results_dir):
    """Create HTML comparing both techniques on same images."""
    # Try to find a pair of result directories for the same image
    bilateral_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_bilateral")) 
                     if os.path.isdir(d)]
    gradient_dirs = [d for d in glob.glob(os.path.join(results_dir, "*_gradient")) 
                    if os.path.isdir(d)]
    
    if not bilateral_dirs or not gradient_dirs:
        return "<p>Insufficient results to create direct comparison.</p>"
    
    html = "<h3>Direct Visual Comparison</h3>"
    html += "<p>Below is a side-by-side comparison of both techniques applied to different samples:</p>"
    
    for b_dir in bilateral_dirs:
        b_name = os.path.basename(b_dir).replace("_bilateral", "")
        b_result = os.path.join(b_dir, "bilateral_final.jpg")
        
        if not os.path.exists(b_result):
            continue
        
        html += f"<h4>Sample: {b_name}</h4>"
        html += "<div class='image-container'>"
        
        # Original images
        ambient_path = os.path.join(results_dir, "..", "sample_images", "bilateral", b_name, "noflash.jpg")
        flash_path = os.path.join(results_dir, "..", "sample_images", "bilateral", b_name, "flash.jpg")
        
        if os.path.exists(ambient_path) and os.path.exists(flash_path):
            html += create_image_html(ambient_path, "Ambient (No Flash)")
            html += create_image_html(flash_path, "Flash")
        
        # Bilateral result
        html += create_image_html(b_result, "Bilateral Filtering Result")
        
        html += "</div>"
    
    for g_dir in gradient_dirs:
        g_name = os.path.basename(g_dir).replace("_gradient", "")
        g_result = os.path.join(g_dir, "gradient_final.jpg")
        
        if not os.path.exists(g_result):
            continue
        
        html += f"<h4>Sample: {g_name}</h4>"
        html += "<div class='image-container'>"
        
        # Original images
        ambient_path = os.path.join(results_dir, "..", "sample_images", "gradient", g_name, "noflash.jpg")
        flash_path = os.path.join(results_dir, "..", "sample_images", "gradient", g_name, "flash.jpg")
        
        if os.path.exists(ambient_path) and os.path.exists(flash_path):
            html += create_image_html(ambient_path, "Ambient (No Flash)")
            html += create_image_html(flash_path, "Flash")
        
        # Gradient result
        html += create_image_html(g_result, "Gradient Domain Result")
        
        html += "</div>"
    
    return html

def main():
    parser = argparse.ArgumentParser(description="Create analysis presentation from flash/no-flash results")
    parser.add_argument("--results_dir", type=str, default="part3_results",
                       help="Directory containing processing results")
    parser.add_argument("--output_file", type=str, default="flash_noflash_analysis.html",
                       help="Output HTML presentation file")
    args = parser.parse_args()
    
    # Create HTML components
    bilateral_results_html = create_bilateral_results_html(args.results_dir)
    gradient_results_html = create_gradient_results_html(args.results_dir)
    bilateral_parameter_html = create_parameter_analysis_html(args.results_dir, "bilateral")
    gradient_parameter_html = create_parameter_analysis_html(args.results_dir, "gradient")
    comparison_html = create_technique_comparison_html(args.results_dir)
    
    # Generate final HTML
    final_html = HTML_TEMPLATE.format(
        bilateral_results_html=bilateral_results_html,
        gradient_results_html=gradient_results_html,
        bilateral_parameter_html=bilateral_parameter_html,
        gradient_parameter_html=gradient_parameter_html,
        comparison_html=comparison_html
    )
    
    # Write HTML to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    print(f"Analysis presentation created: {os.path.abspath(args.output_file)}")
    print("Open this file in a web browser to view the analysis.")

if __name__ == "__main__":
    main() 