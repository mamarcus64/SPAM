import os
import sys
import cv2
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data.image
from src.point_extraction import PointExtractor
from src.spam import Spam

original_checkpoint = '/scratch1/mjma/SPAN/interspeech_spam/segment-anything-2/checkpoints/sam2_hiera_tiny.pt'
config_file = 'sam2_hiera_t.yaml'
spam_checkpoint = '/scratch1/mjma/SPAN/interspeech_spam/spam_finetuning/saved_models/tiny/checkpoints/step_13200.pt'

spam = Spam(config_file, original_checkpoint, spam_checkpoint).to('cuda')

point_extractor = PointExtractor('best_unet.pth')

image = data.image.Image('test_image.png')

points = point_extractor.extract_points(image.image)

print(points)

# SPAM expects 3-channel (RGB), currently grayscale
spam_image = cv2.cvtColor(image.image, cv2.COLOR_GRAY2RGB)
# spam_image = spam_image.astype('float16') / 255.0

assert spam_image.shape == (512, 512, 3)
assert spam_image.dtype == np.uint8

masks = spam.predict_frame(spam_image, points)

stacked_masks = np.stack(masks, axis=0)

data.image.save_image_with_masks(image.image, stacked_masks, 'result.png')