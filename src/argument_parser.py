import argparse
import os
import sys
from pathlib import Path

# Supported formats according to OpenCV documentation
IMAGE_FORMATS = {'.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.tiff', '.tif'}
VIDEO_FORMATS = {'.avi', '.mp4', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.3gp', '.3g2'}

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
MRI Image Segmentation Tool

This tool accepts a single input file (image or video) and performs automatic segmentation.
The input file must be compatible with OpenCV. Supported formats include:

Images: {}
Videos: {}

The output can be saved in one or more formats: rendered pictures, NumPy arrays, or Run Length Encodings (RLE).
        """.format(', '.join(sorted(IMAGE_FORMATS)), ', '.join(sorted(VIDEO_FORMATS))),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'file',
        help='Path to an input image or video file. Must be a format supported by OpenCV.'
    )

    parser.add_argument(
        '--output-type', '-t',
        nargs='+',
        choices=['image', 'numpy', 'RLE'],
        default=['image'],
        help=(
            'Specify one or more output types for the segmentation results.\n'
            'Choose from: "image", "numpy", "RLE".\n'
            'Example: --output-type image numpy'
        )
    )

    parser.add_argument(
        '--output-folder', '-o',
        default=os.getcwd(),
        help='Directory to save output files. Defaults to the current working directory.'
    )
    
    parser.add_argument(
        '--resize-to-512',
        action='store_true',
        help=(
            'Images are scaled to 512x512 pixels during segmentation and then scaled back to '
            'the original image size afterwards.\nSet this flag to keep the images as 512x512.'
        )
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help=(
            'Device to run segmentation on.\n'
            '"cpu" forces CPU usage.\n'
            '"cuda" (default) forces GPU usage (fails if unavailable).'
        )
    )

    args = parser.parse_args()

    # Validate and determine input type
    ext = os.path.splitext(args.file)[-1].lower()
    args.filestem = Path(args.file).stem

    if ext in IMAGE_FORMATS:
        args.input_type = 'image'
    elif ext in VIDEO_FORMATS:
        args.input_type = 'video'
    else:
        parser.error(
            f"Unsupported file extension '{ext}'.\n"
            f"Supported image formats: {', '.join(sorted(IMAGE_FORMATS))}\n"
            f"Supported video formats: {', '.join(sorted(VIDEO_FORMATS))}"
        )

    args.extension = ext

    return args
