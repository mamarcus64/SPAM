import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data.image
from src.point_extraction import PointExtractor
from src.spam import Spam
from src.argument_parser import parse_args
import src.config as cfg

args = parse_args()

spam = Spam(cfg.config_file, cfg.original_checkpoint, cfg.spam_checkpoint, device=args.device).to(args.device)
point_extractor = PointExtractor(cfg.unet_file)

# bad naming scheme - but there's a difference between an Image object and just the numpy array of a loaded image
def segment_image(original_image):
    image = data.image.Image(original_image=original_image)
    points = point_extractor.extract_points(image.image)

    # SPAM expects 3-channel (RGB), currently grayscale
    spam_image = cv2.cvtColor(image.image, cv2.COLOR_GRAY2RGB)
    # spam_image = spam_image.astype('float16') / 255.0

    assert spam_image.shape == (512, 512, 3)
    assert spam_image.dtype == np.uint8

    masks = spam.predict_frame(spam_image, points)
    image.set_masks_numpy(masks)
    
    return image
    
if args.input_type == 'image':
    raw_image = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
    image = segment_image(raw_image)
    
    if 'image' in args.output_type:
        location = os.path.join(args.output_folder, f'{args.filestem}_segmented{args.extension}')
        blended = image.output_image_with_masks(resize_to_original=not args.resize_to_512)
        cv2.imwrite(location, blended)
        
    if 'numpy' in args.output_type:
        location = os.path.join(args.output_folder, f'{args.filestem}_segmented.npy')
        stacked_masks = image.output_masks_numpy(resize_to_original=not args.resize_to_512)
        np.save(location, stacked_masks)
        
    if 'RLE' in args.output_type:
        location = os.path.join(args.output_folder, f'{args.filestem}_segmented.json')
        rle_data = image.output_masks_rle(resize_to_original=not args.resize_to_512)
        json.dump(rle_data, open(location, 'w'))
        
    
    
if args.input_type == 'video':
    capture = cv2.VideoCapture(args.file)
    fps = capture.get(cv2.CAP_PROP_FPS)
    # rounding to one decimal (throws error if too precise)
    fps = round(fps, 1)
    
    raw_frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_frames.append(gray_frame)
    capture.release()

    images = []
    
    for frame in tqdm(raw_frames, desc="Processing video frame"):
        images.append(segment_image(frame))
        
    if 'image' in args.output_type:
        # always saves as mp4
        location = os.path.join(args.output_folder, f'{args.filestem}_segmented.mp4')
        
        all_frames = []
        
        for image in images:
            blended = image.output_image_with_masks(resize_to_original=not args.resize_to_512)
            all_frames.append(blended)
            
        # write to video
        height, width = all_frames[0].shape[:2]
        is_color = len(all_frames[0].shape) == 3
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(location, fourcc, fps, (width, height), isColor=is_color)
        
        for frame in all_frames:
            if not is_color and frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        out.release()
        
    if 'numpy' in args.output_type:
        location = os.path.join(args.output_folder, f'{args.filestem}_segmented.npy')
        
        all_stacked_masks = []
        
        for image in images:
            stacked_masks = image.output_masks_numpy(resize_to_original=not args.resize_to_512)
            all_stacked_masks.append(stacked_masks)
            
        np.save(location, np.stack(all_stacked_masks, axis=0))
        
    if 'RLE' in args.output_type:
        location = os.path.join(args.output_folder, f'{args.filestem}_segmented.json')
        
        all_rle_data = {}
        
        for i, image in enumerate(images):
            rle_data = image.output_masks_rle(resize_to_original=not args.resize_to_512)
            all_rle_data[f'frame_{i}'] = rle_data
            
        json.dump(all_rle_data, open(location, 'w'))