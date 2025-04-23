#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash/No-Flash Photography Sample Image Downloader

This script downloads sample flash/no-flash image pairs from public repositories
for demonstration purposes. It downloads pairs suitable for:
1. Bilateral filtering (denoising)
2. Gradient domain processing (fusion)

Usage:
    python download_sample_pairs.py
"""

import os
import urllib.request
import shutil
import zipfile
import tarfile
import argparse
import ssl

# Disable SSL certificate verification (for some environments)
ssl._create_default_https_context = ssl._create_unverified_context

# URLs for sample flash/no-flash image pairs
FLASH_NOFLASH_SAMPLES = {
    # Bilateral filtering examples
    "bilateral": [
        {
            "name": "toys",
            "flash_url": "https://github.com/ceciliazheng/flashNoflash/raw/master/images/toys_flash.jpg",
            "noflash_url": "https://github.com/ceciliazheng/flashNoflash/raw/master/images/toys_noflash.jpg",
            "description": "Toy collection in a dimly lit environment"
        },
        {
            "name": "figurine",
            "flash_url": "https://raw.githubusercontent.com/neycyanshi/Denoising_Flash_No-Flash/master/images/figure/figure1_flash.jpg",
            "noflash_url": "https://raw.githubusercontent.com/neycyanshi/Denoising_Flash_No-Flash/master/images/figure/figure1_noflash.jpg",
            "description": "Figurine in a dark setting, good for denoising"
        }
    ],
    
    # Gradient domain examples
    "gradient": [
        {
            "name": "plant",
            "flash_url": "https://raw.githubusercontent.com/neycyanshi/Denoising_Flash_No-Flash/master/images/plant/plant_flash.jpg",
            "noflash_url": "https://raw.githubusercontent.com/neycyanshi/Denoising_Flash_No-Flash/master/images/plant/plant_noflash.jpg",
            "description": "Plant with mixed specular and matte surfaces"
        },
        {
            "name": "sculpture",
            "flash_url": "https://raw.githubusercontent.com/neycyanshi/Denoising_Flash_No-Flash/master/images/furn/furn_flash.jpg",
            "noflash_url": "https://raw.githubusercontent.com/neycyanshi/Denoising_Flash_No-Flash/master/images/furn/furn_noflash.jpg", 
            "description": "Sculpture with both specular highlights and shadows"
        }
    ]
}

def download_file(url, filename):
    """Download a file from a URL to a local destination."""
    try:
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_image_pairs(output_dir="sample_images"):
    """Download sample flash/no-flash image pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    for technique, pairs in FLASH_NOFLASH_SAMPLES.items():
        # Create subdirectory for each technique
        technique_dir = os.path.join(output_dir, technique)
        os.makedirs(technique_dir, exist_ok=True)
        
        for pair in pairs:
            # Create subdirectory for each image pair
            pair_dir = os.path.join(technique_dir, pair["name"])
            os.makedirs(pair_dir, exist_ok=True)
            
            # Download flash image
            flash_filename = os.path.join(pair_dir, "flash.jpg")
            if not os.path.exists(flash_filename):
                print(f"Downloading {pair['name']} flash image...")
                if download_file(pair["flash_url"], flash_filename):
                    print(f"  Saved to {flash_filename}")
                else:
                    print(f"  Failed to download flash image for {pair['name']}")
            else:
                print(f"{flash_filename} already exists, skipping download")
            
            # Download no-flash image
            noflash_filename = os.path.join(pair_dir, "noflash.jpg")
            if not os.path.exists(noflash_filename):
                print(f"Downloading {pair['name']} no-flash image...")
                if download_file(pair["noflash_url"], noflash_filename):
                    print(f"  Saved to {noflash_filename}")
                else:
                    print(f"  Failed to download no-flash image for {pair['name']}")
            else:
                print(f"{noflash_filename} already exists, skipping download")
            
            # Create a description file
            with open(os.path.join(pair_dir, "description.txt"), "w") as f:
                f.write(f"Sample: {pair['name']}\n")
                f.write(f"Technique: {technique}\n")
                f.write(f"Description: {pair['description']}\n")
    
    print("\nDownload summary:")
    for technique, pairs in FLASH_NOFLASH_SAMPLES.items():
        print(f"\n{technique.upper()} technique examples:")
        for pair in pairs:
            print(f"  - {pair['name']}: {pair['description']}")
            pair_dir = os.path.join(output_dir, technique, pair["name"])
            flash_file = os.path.join(pair_dir, "flash.jpg")
            noflash_file = os.path.join(pair_dir, "noflash.jpg")
            if os.path.exists(flash_file) and os.path.exists(noflash_file):
                print(f"    ✓ Both images downloaded successfully")
            else:
                missing = []
                if not os.path.exists(flash_file):
                    missing.append("flash")
                if not os.path.exists(noflash_file):
                    missing.append("no-flash")
                print(f"    ✗ Missing {', '.join(missing)} image(s)")

def main():
    parser = argparse.ArgumentParser(description="Download sample flash/no-flash image pairs")
    parser.add_argument("--output_dir", type=str, default="sample_images",
                       help="Directory to save downloaded images")
    args = parser.parse_args()
    
    print("Flash/No-Flash Photography Sample Image Downloader")
    print("==================================================")
    print(f"Images will be downloaded to: {args.output_dir}")
    
    download_image_pairs(args.output_dir)
    
    print("\nDownload complete!")
    print("You can now use these images with the demo scripts.")
    print("\nFor bilateral filtering examples:")
    print("  python demo_bilateral.py --data_dir sample_images/bilateral/toys --flash_name flash.jpg --noflash_name noflash.jpg")
    print("\nFor gradient domain processing examples:")
    print("  python demo_gradient_domain.py --data_dir sample_images/gradient/plant --image_name flash.jpg")

if __name__ == "__main__":
    main() 