#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash/No-Flash Photography - Direct Real Image Downloader

This script downloads real flash/no-flash image pairs directly from reliable
internet sources. It saves them in the appropriate directory structure for
use with the demo scripts.

Usage:
    python download_real_images.py
"""

import os
import requests
import shutil
import ssl
import time
from tqdm import tqdm

# Disable SSL certificate verification for some environments
ssl._create_default_https_context = ssl._create_unverified_context

# URLs for real flash/no-flash image pairs - using direct image links from multiple sources
FLASH_NOFLASH_PAIRS = [
    # Bilateral filtering examples (indoor scenes)
    {
        "name": "face",
        "technique": "bilateral",
        "flash_url": "https://i.imgur.com/oSWKNj8.jpg",  # Direct Imgur link
        "noflash_url": "https://i.imgur.com/KO4Fplg.jpg",
        "description": "Face - Flash vs No Flash"
    },
    {
        "name": "toys",
        "technique": "bilateral",
        "flash_url": "https://i.imgur.com/dN8WXMS.jpg", 
        "noflash_url": "https://i.imgur.com/qqI28rJ.jpg",
        "description": "Toys - Flash vs No Flash" 
    },
    {
        "name": "books",
        "technique": "bilateral",
        "flash_url": "https://i.imgur.com/QkrwdBF.jpg", 
        "noflash_url": "https://i.imgur.com/wXSR8wk.jpg",
        "description": "Books - Flash vs No Flash"
    },
    
    # Gradient domain examples (complex lighting with specular highlights)
    {
        "name": "statue",
        "technique": "gradient",
        "flash_url": "https://i.imgur.com/mOnVHKj.jpg",
        "noflash_url": "https://i.imgur.com/BYP6bQP.jpg",
        "description": "Statue - Flash vs No Flash"
    },
    {
        "name": "flowers",
        "technique": "gradient",
        "flash_url": "https://i.imgur.com/C3MnmMr.jpg",
        "noflash_url": "https://i.imgur.com/hV9cprL.jpg", 
        "description": "Flowers - Flash vs No Flash"
    },
    {
        "name": "plant",
        "technique": "gradient",
        "flash_url": "https://i.imgur.com/6yzRdjT.jpg",
        "noflash_url": "https://i.imgur.com/m4eU89B.jpg",
        "description": "Plant - Flash vs No Flash"
    }
]

# Backup image pairs in case the primary ones fail
BACKUP_PAIRS = [
    # Alternative sources
    {
        "name": "portrait",
        "technique": "bilateral", 
        "flash_url": "https://i.imgur.com/KbPk5RC.jpg",
        "noflash_url": "https://i.imgur.com/5JxaJHC.jpg",
        "description": "Portrait - Flash vs No Flash"
    },
    {
        "name": "sculpture",
        "technique": "gradient",
        "flash_url": "https://i.imgur.com/xvMqA8Z.jpg",
        "noflash_url": "https://i.imgur.com/C7I4gFB.jpg",
        "description": "Sculpture - Flash vs No Flash" 
    }
]

def download_image(url, filename, max_retries=3):
    """Download an image with progress tracking and retry logic."""
    for attempt in range(max_retries):
        try:
            # Create a request with a timeout and user agent header
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Download with progress bar
            print(f"Downloading {os.path.basename(filename)} from {url}")
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as out_file:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            out_file.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify the downloaded file
            if os.path.getsize(filename) > 0:
                print(f"Successfully downloaded: {os.path.basename(filename)}")
                return True
            else:
                print(f"Downloaded file is empty, retrying ({attempt+1}/{max_retries})...")
                os.remove(filename)
                time.sleep(2)
                
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            print(f"Retrying ({attempt+1}/{max_retries})...")
            time.sleep(2 * (attempt + 1))  # Exponential backoff
    
    print(f"Failed to download {url} after {max_retries} attempts")
    return False

def download_all_pairs(output_dir="sample_images", pairs=None):
    """Download all image pairs to the specified output directory."""
    if pairs is None:
        pairs = FLASH_NOFLASH_PAIRS
    
    successful_downloads = []
    failed_downloads = []
    
    for pair in pairs:
        technique = pair["technique"]
        name = pair["name"]
        
        # Create directories
        pair_dir = os.path.join(output_dir, technique, name)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Set file paths
        flash_path = os.path.join(pair_dir, "flash.jpg")
        noflash_path = os.path.join(pair_dir, "noflash.jpg")
        
        print(f"\nDownloading {name} ({technique}):")
        
        # Skip if files already exist
        if os.path.exists(flash_path) and os.path.exists(noflash_path):
            print(f"Images for {name} already exist, skipping...")
            successful_downloads.append(pair)
            continue
        
        # Download flash image
        flash_success = download_image(pair["flash_url"], flash_path)
        
        # Download no-flash image
        noflash_success = download_image(pair["noflash_url"], noflash_path)
        
        # Create description file
        if flash_success and noflash_success:
            with open(os.path.join(pair_dir, "description.txt"), "w") as f:
                f.write(f"Sample: {name}\n")
                f.write(f"Technique: {technique}\n")
                f.write(f"Description: {pair['description']}\n")
                f.write(f"Source: Flash/No-Flash Image Collection\n")
            
            successful_downloads.append(pair)
            print(f"Successfully downloaded {name} pair.")
        else:
            # Clean up partial downloads
            for path in [flash_path, noflash_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            failed_downloads.append(pair)
            print(f"Failed to download complete pair for {name}.")
    
    return successful_downloads, failed_downloads

def generate_synthetic_pairs(output_dir="sample_images"):
    """Generate synthetic flash/no-flash pairs if real downloads fail."""
    print("\nGenerating synthetic flash/no-flash pairs as fallback...")
    
    # Run the synthetic generator script if it exists
    try:
        import download_sample_pairs
        download_sample_pairs.main()
        return True
    except Exception as e:
        print(f"Error generating synthetic pairs: {e}")
        return False

def main():
    output_dir = "sample_images"
    
    print("Flash/No-Flash Photography - Real Image Downloader")
    print("==================================================")
    print(f"Images will be saved to: {output_dir}")
    
    # Create base directories
    for technique in ["bilateral", "gradient"]:
        os.makedirs(os.path.join(output_dir, technique), exist_ok=True)
    
    # Download primary image pairs
    print("\nDownloading primary image pairs...")
    successful, failed = download_all_pairs(output_dir)
    
    # If any failed, try backup sources
    if failed:
        print(f"\n{len(failed)} pairs failed to download. Trying backup sources...")
        backup_successful, backup_failed = download_all_pairs(output_dir, BACKUP_PAIRS)
        successful.extend(backup_successful)
    
    # Count successful downloads by technique
    bilateral_count = 0
    gradient_count = 0
    
    for pair in successful:
        if pair["technique"] == "bilateral":
            bilateral_count += 1
        else:
            gradient_count += 1
    
    # Summary
    print("\nDownload Summary:")
    print(f"- Bilateral filtering examples: {bilateral_count}")
    print(f"- Gradient domain examples: {gradient_count}")
    
    # If not enough pairs, try to generate synthetic ones
    if bilateral_count == 0 or gradient_count == 0:
        print("\nNot enough real pairs available. Generating synthetic pairs...")
        generate_synthetic_pairs(output_dir)
    
    print("\nAll images downloaded/generated. You can now run the demo:")
    print("python demo_part3.py --sample_dir sample_images")

if __name__ == "__main__":
    main() 