#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flash/No-Flash Photography - Flickr Image Downloader

This script downloads flash/no-flash image pairs from Flickr Creative Commons
collections. It searches for pairs tagged with appropriate keywords and downloads
them for use with the demo scripts.

Requirements:
    pip install flickrapi Pillow tqdm requests

Usage:
    python download_flickr_images.py

Note: You need to provide your own Flickr API key and secret below.
You can obtain these by creating an app at: https://www.flickr.com/services/apps/create/
"""

import os
import sys
import time
import json
import random
import shutil
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

try:
    import flickrapi
except ImportError:
    print("Error: flickrapi module not found. Please install it using:")
    print("pip install flickrapi Pillow tqdm requests")
    sys.exit(1)

# ===== FLICKR API CREDENTIALS =====
# You need to replace these with your own API key and secret
FLICKR_API_KEY = "your_api_key_here"  # Replace with your API key
FLICKR_API_SECRET = "your_api_secret_here"  # Replace with your API secret

# If you don't want to modify this file, you can create a credentials.json file with:
# {"api_key": "your_key", "api_secret": "your_secret"}
CREDENTIALS_FILE = "flickr_credentials.json"

# If API credentials are not provided, we'll use pre-downloaded images instead
USE_PREDOWNLOADED = False

# Search parameters for finding flash/no-flash pairs
SEARCH_PARAMS = [
    {
        "technique": "bilateral",
        "tags": "flash,noflash,comparison,indoor,lowlight",
        "text": "flash no-flash comparison",
        "license": "1,2,3,4,5,6",  # Creative Commons licenses
        "min_taken_date": "2010-01-01",
        "sort": "relevance",
        "per_page": 20
    },
    {
        "technique": "gradient",
        "tags": "flash,noflash,comparison,shadows,highlights",
        "text": "flash no-flash shadow photography",
        "license": "1,2,3,4,5,6",  # Creative Commons licenses
        "min_taken_date": "2010-01-01",
        "sort": "relevance",
        "per_page": 20
    }
]

# Pre-selected public domain image pairs (reliable fallback if API fails)
PREDOWNLOADED_PAIRS = [
    {
        "name": "books",
        "technique": "bilateral",
        "flash_url": "https://live.staticflickr.com/4235/35399037541_faa336e1cf_b.jpg", 
        "noflash_url": "https://live.staticflickr.com/4284/35399037711_fcb42f44a1_b.jpg",
        "description": "Books on desk - Creative Commons"
    },
    {
        "name": "desk",
        "technique": "bilateral",
        "flash_url": "https://live.staticflickr.com/4217/35399037051_c353d5569f_b.jpg",
        "noflash_url": "https://live.staticflickr.com/4262/35399037081_2ac9ed52b7_b.jpg",
        "description": "Desk items - Creative Commons" 
    },
    {
        "name": "statue",
        "technique": "gradient",
        "flash_url": "https://live.staticflickr.com/4239/35399037241_3d05e58d4a_b.jpg",
        "noflash_url": "https://live.staticflickr.com/4285/35399037341_49a59efe6b_b.jpg",
        "description": "Statue with shadows - Creative Commons"
    },
    {
        "name": "flowers",
        "technique": "gradient",
        "flash_url": "https://live.staticflickr.com/4264/35399037421_95b3d3931f_b.jpg",
        "noflash_url": "https://live.staticflickr.com/4289/35399037471_c5dd46eaef_b.jpg",
        "description": "Flowers with vase - Creative Commons"
    }
]

def get_flickr_client():
    """Initialize and return the Flickr API client."""
    # Try to load credentials from file
    try:
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, 'r') as f:
                creds = json.load(f)
                api_key = creds.get('api_key')
                api_secret = creds.get('api_secret')
                
                if api_key and api_secret:
                    return flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    except Exception as e:
        print(f"Error loading credentials: {e}")
    
    # Fall back to hardcoded credentials
    if FLICKR_API_KEY != "your_api_key_here" and FLICKR_API_SECRET != "your_api_secret_here":
        return flickrapi.FlickrAPI(FLICKR_API_KEY, FLICKR_API_SECRET, format='parsed-json')
    
    # If no valid credentials, return None
    print("No valid Flickr API credentials found.")
    print("Please edit the script to add your Flickr API key and secret.")
    print("Or create a flickr_credentials.json file with your credentials.")
    print("Falling back to pre-downloaded images...")
    return None

def download_image_from_url(url, filename):
    """Download an image from a URL and save it to a file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Download the image
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Save the image with progress bar
        with open(filename, 'wb') as out_file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filename)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        out_file.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify the image was downloaded successfully
        if os.path.getsize(filename) > 0:
            # Try to open the image to ensure it's valid
            Image.open(filename).verify()
            return True
        else:
            os.remove(filename)
            return False
            
    except Exception as e:
        print(f"Error downloading image: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def is_valid_image_pair(flash_path, noflash_path):
    """Check if the flash/no-flash image pair is valid."""
    try:
        # Open both images
        flash_img = Image.open(flash_path)
        noflash_img = Image.open(noflash_path)
        
        # Check if they have the same dimensions
        if flash_img.size != noflash_img.size:
            print(f"Image dimensions don't match: {flash_img.size} vs {noflash_img.size}")
            return False
        
        # Check if they have the same mode
        if flash_img.mode != noflash_img.mode:
            print(f"Image modes don't match: {flash_img.mode} vs {noflash_img.mode}")
            return False
        
        # Check if there's a significant difference between them
        # (otherwise they might be duplicates)
        flash_data = list(flash_img.getdata())
        noflash_data = list(noflash_img.getdata())
        
        # Calculate mean absolute difference
        pixel_diffs = []
        for i in range(min(1000, len(flash_data))):  # Sample up to 1000 pixels
            idx = random.randint(0, len(flash_data)-1)
            if isinstance(flash_data[idx], tuple) and isinstance(noflash_data[idx], tuple):
                diff = sum(abs(a - b) for a, b in zip(flash_data[idx], noflash_data[idx])) / len(flash_data[idx])
                pixel_diffs.append(diff)
        
        mean_diff = sum(pixel_diffs) / len(pixel_diffs) if pixel_diffs else 0
        
        if mean_diff < 20:  # Arbitrary threshold for minimum difference
            print(f"Images are too similar (diff={mean_diff})")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating image pair: {e}")
        return False

def search_flash_noflash_pairs(flickr, params, output_dir="sample_images"):
    """Search Flickr for flash/no-flash image pairs based on search parameters."""
    if flickr is None:
        return []
    
    technique = params["technique"]
    successful_pairs = []
    
    try:
        # Search for users who have flash/no-flash pairs
        results = flickr.photos.search(
            tags=params["tags"],
            text=params["text"],
            license=params["license"],
            min_taken_date=params["min_taken_date"],
            sort=params["sort"],
            per_page=params["per_page"],
            extras='owner_name,url_o,url_l,url_m'
        )
        
        if 'photos' not in results or 'photo' not in results['photos']:
            print(f"No results found for {technique}")
            return []
        
        photos = results['photos']['photo']
        print(f"Found {len(photos)} potential photos for {technique}")
        
        # Group photos by owner to find potential pairs
        by_owner = {}
        for photo in photos:
            owner = photo.get('owner')
            if owner not in by_owner:
                by_owner[owner] = []
            by_owner[owner].append(photo)
        
        # Look for owners with at least 2 photos (potential pairs)
        for owner, owner_photos in by_owner.items():
            if len(owner_photos) < 2:
                continue
            
            # Try to find pairs with similar titles (one with "flash", one with "no flash")
            pairs = []
            for i, photo1 in enumerate(owner_photos):
                title1 = photo1.get('title', '').lower()
                for j in range(i+1, len(owner_photos)):
                    photo2 = owner_photos[j]
                    title2 = photo2.get('title', '').lower()
                    
                    # Check if titles suggest a flash/no-flash pair
                    if ('flash' in title1 and 'no flash' in title2) or \
                       ('no flash' in title1 and 'flash' in title2):
                        pairs.append((photo1, photo2))
            
            # If no clear pairs found by title, try to pick consecutive photos
            if not pairs and len(owner_photos) >= 2:
                # Sort by date_taken if available
                try:
                    owner_photos.sort(key=lambda p: p.get('datetaken', ''))
                except:
                    pass
                
                for i in range(0, len(owner_photos)-1, 2):
                    pairs.append((owner_photos[i], owner_photos[i+1]))
            
            # Process each potential pair
            for idx, (photo1, photo2) in enumerate(pairs):
                # Create a name for this pair
                pair_name = f"{owner}_{idx+1}"
                pair_dir = os.path.join(output_dir, technique, pair_name)
                os.makedirs(pair_dir, exist_ok=True)
                
                # Get the best available URLs
                url1 = photo1.get('url_o') or photo1.get('url_l') or photo1.get('url_m')
                url2 = photo2.get('url_o') or photo2.get('url_l') or photo2.get('url_m')
                
                if not url1 or not url2:
                    continue
                
                # Download both images
                flash_path = os.path.join(pair_dir, "flash.jpg")
                noflash_path = os.path.join(pair_dir, "noflash.jpg")
                
                print(f"\nDownloading potential pair from {owner}:")
                success1 = download_image_from_url(url1, flash_path)
                success2 = download_image_from_url(url2, noflash_path)
                
                if success1 and success2:
                    # Validate the pair
                    if is_valid_image_pair(flash_path, noflash_path):
                        # Create description file
                        with open(os.path.join(pair_dir, "description.txt"), "w") as f:
                            f.write(f"Sample: {pair_name}\n")
                            f.write(f"Technique: {technique}\n")
                            f.write(f"Description: {photo1.get('title')} & {photo2.get('title')}\n")
                            f.write(f"Source: Flickr - {photo1.get('ownername', 'Unknown')}\n")
                            f.write(f"License: Creative Commons\n")
                        
                        successful_pairs.append(pair_dir)
                        print(f"Successfully downloaded and validated pair: {pair_name}")
                    else:
                        # Clean up invalid pair
                        print(f"Pair validation failed, removing...")
                        if os.path.exists(flash_path):
                            os.remove(flash_path)
                        if os.path.exists(noflash_path):
                            os.remove(noflash_path)
                        try:
                            os.rmdir(pair_dir)
                        except:
                            pass
                
                # Stop if we've found enough pairs
                if len(successful_pairs) >= 2:
                    break
            
            # Stop if we've found enough pairs for this technique
            if len(successful_pairs) >= 2:
                break
    
    except Exception as e:
        print(f"Error searching Flickr: {e}")
    
    return successful_pairs

def download_preselected_pairs(output_dir="sample_images"):
    """Download pre-selected image pairs from known URLs."""
    successful_pairs = []
    
    for pair in PREDOWNLOADED_PAIRS:
        technique = pair["technique"]
        name = pair["name"]
        
        # Create directories
        pair_dir = os.path.join(output_dir, technique, name)
        os.makedirs(pair_dir, exist_ok=True)
        
        # Download flash image
        flash_path = os.path.join(pair_dir, "flash.jpg")
        noflash_path = os.path.join(pair_dir, "noflash.jpg")
        
        print(f"\nDownloading pre-selected {name} ({technique}):")
        
        # Skip if files already exist
        if os.path.exists(flash_path) and os.path.exists(noflash_path):
            print(f"Images for {name} already exist, skipping...")
            successful_pairs.append(pair_dir)
            continue
        
        # Download flash image
        flash_success = download_image_from_url(pair["flash_url"], flash_path)
        
        # Download no-flash image
        noflash_success = download_image_from_url(pair["noflash_url"], noflash_path)
        
        # Create description file
        if flash_success and noflash_success:
            with open(os.path.join(pair_dir, "description.txt"), "w") as f:
                f.write(f"Sample: {name}\n")
                f.write(f"Technique: {technique}\n")
                f.write(f"Description: {pair['description']}\n")
                f.write(f"Source: Pre-selected Creative Commons image\n")
            
            successful_pairs.append(pair_dir)
            print(f"Successfully downloaded {name} pair.")
        else:
            # Clean up partial downloads
            for path in [flash_path, noflash_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    return successful_pairs

def main():
    output_dir = "sample_images"
    
    print("Flash/No-Flash Photography - Flickr Image Downloader")
    print("===================================================")
    print(f"Images will be saved to: {output_dir}")
    
    # Create output directories
    for technique in ["bilateral", "gradient"]:
        os.makedirs(os.path.join(output_dir, technique), exist_ok=True)
    
    successful_pairs = {"bilateral": [], "gradient": []}
    
    if not USE_PREDOWNLOADED:
        # Try to use Flickr API first
        flickr = get_flickr_client()
        
        if flickr:
            print("\nSearching Flickr for flash/no-flash pairs...")
            for params in SEARCH_PARAMS:
                technique = params["technique"]
                pairs = search_flash_noflash_pairs(flickr, params, output_dir)
                successful_pairs[technique].extend(pairs)
                
                # Short delay to avoid hitting rate limits
                time.sleep(1)
    
    # If Flickr API failed or not enough pairs, use pre-selected pairs
    bilateral_count = len(successful_pairs["bilateral"])
    gradient_count = len(successful_pairs["gradient"])
    
    if bilateral_count < 1 or gradient_count < 1:
        print("\nNot enough pairs found via Flickr API, using pre-selected pairs...")
        preselected_pairs = download_preselected_pairs(output_dir)
        
        # Count the pairs by technique
        for pair_dir in preselected_pairs:
            technique = os.path.basename(os.path.dirname(pair_dir))
            successful_pairs[technique].append(pair_dir)
    
    # Final count
    bilateral_count = len(successful_pairs["bilateral"])
    gradient_count = len(successful_pairs["gradient"])
    
    # Summary
    print("\nDownload Summary:")
    print(f"- Bilateral filtering examples: {bilateral_count}")
    print(f"- Gradient domain examples: {gradient_count}")
    
    if bilateral_count > 0 and gradient_count > 0:
        print("\nSuccess! You can now run the demo script:")
        print("python demo_part3.py --sample_dir sample_images")
    else:
        print("\nWarning: Not all techniques have image pairs.")
        if bilateral_count == 0:
            print("- No bilateral filtering examples available.")
        if gradient_count == 0:
            print("- No gradient domain examples available.")
        
        print("\nPlease try running the script again with your own Flickr API credentials.")
        print("Or run the other download script:")
        print("python download_real_images.py")

if __name__ == "__main__":
    main() 