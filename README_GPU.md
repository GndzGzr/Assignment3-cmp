# Flash/No-Flash Photography with GPU Acceleration

This extension of the bilateral filtering implementation for flash/no-flash photography adds GPU acceleration using OpenCV's CUDA module. It can provide significant speedups compared to the CPU-only implementation, especially for larger images and more complex filtering operations.

## GPU Requirements

This implementation requires:
- NVIDIA GPU with CUDA support
- OpenCV built with CUDA support (`opencv-contrib-python` typically includes this)
- CUDA Toolkit installed on your system

## Overview

The GPU-accelerated implementation follows the same approach as the CPU version but leverages the GPU for computationally intensive operations:

1. Basic bilateral filtering uses `cv2.cuda.bilateralFilter`
2. Joint bilateral filtering still uses CPU (OpenCV CUDA doesn't have a direct equivalent)
3. Detail transfer arithmetic operations use `cv2.cuda.add`, `cv2.cuda.divide`, and `cv2.cuda.multiply`
4. Shadow and specularity masking uses `cv2.cuda.compare` and `cv2.cuda.bitwise_or`
5. Final image fusion uses `cv2.cuda.multiply` and `cv2.cuda.add`

## Performance

You can expect speedups in the range of 2-10x depending on:
- Image size (larger images generally see better speedups)
- GPU capability
- Parameter settings (larger kernel sizes benefit more from GPU acceleration)

## Usage

### Basic Demo with GPU Acceleration

To run the demo with GPU acceleration:

```bash
python demo_bilateral_filter_gpu.py --data_dir data/camera --image_name cave-flash.jpg
```

This will process the image using GPU acceleration if available and save the results in the `results_gpu` directory.

### Compare GPU vs CPU Performance

To compare GPU and CPU performance:

```bash
python demo_bilateral_filter_gpu.py --data_dir data/camera --image_name cave-flash.jpg --compare_cpu
```

This will process the image using both GPU and CPU implementations, measure performance, and save a comparison of results.

### Disable GPU Acceleration

If you want to run on CPU even if a GPU is available:

```bash
python demo_bilateral_filter_gpu.py --data_dir data/camera --image_name cave-flash.jpg --no_gpu
```

### Batch Processing with GPU Acceleration

To process all images in the dataset with GPU acceleration:

```bash
python batch_processing_gpu.py --data_dir data/camera
```

## Additional Options

All the original options from the CPU implementation are available:

- `--sigma_s_basic` - Spatial sigma for basic bilateral filtering (default: 8.0)
- `--sigma_r_basic` - Range sigma for basic bilateral filtering (default: 0.1)
- `--sigma_s_joint` - Spatial sigma for joint bilateral filtering (default: 8.0)
- `--sigma_r_joint` - Range sigma for joint bilateral filtering (default: 0.1)
- `--epsilon` - Small constant for detail transfer (default: 0.02)
- `--shadow_thresh` - Threshold for shadow detection (default: 0.1)
- `--spec_thresh` - Threshold for specularity detection (default: 0.9)

New GPU-specific options:
- `--no_gpu` - Disable GPU acceleration
- `--compare_cpu` - Compare performance with CPU implementation

## Implementation Details

The GPU implementation:

1. Checks for CUDA availability at startup and falls back to CPU if not available
2. Transfers images to and from the GPU as needed
3. Uses CUDA-accelerated OpenCV functions where available
4. Provides a seamless switch between CPU and GPU processing

### Limitations

- Joint bilateral filtering is still performed on the CPU due to lack of a direct CUDA implementation in OpenCV
- There is some overhead in transferring data between CPU and GPU memory
- Some minor numerical differences may exist between CPU and GPU results due to implementation differences

## References

- OpenCV CUDA Module Documentation: https://docs.opencv.org/master/d0/d05/group__cudaarithm.html
- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/ 