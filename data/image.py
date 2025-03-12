import numpy as np
import cv2
from enum import Enum
import random
import json

class Region(Enum):
    LOWER = 0
    UPPER = 1
    BACK = 2

class Image:
    
    RESIZED_WIDTH, RESIZED_HEIGHT = 512, 512

    def __init__(self, image_path, do_denoise=True):
        
        # load original image
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.original_dimensions = self.original_image.shape
        
        # preprocessing
        if self.original_dimensions == (Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT):
            self.image = self.original_image
        else:
            self.image = cv2.resize(self.original_image, (Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT), interpolation=cv2.INTER_LINEAR)
        if do_denoise:
            self.image = cv2.fastNlMeansDenoising(
                self.image, None, h=11, templateWindowSize=7, searchWindowSize=21
            )
        
        # create masks
        self.masks = {}
        for region in Region:
            # note to self: make sure to stay consistent with np.uint8!
            self.masks[region.value] = np.zeros(shape=(Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT), dtype=np.uint8)

    def save_masks_rle(self, location):
        """
        Using COCO's version of run-length encoding.
        Note that this flattens column-wise rather than row-wise.
        """
        rle_data = {
            'width': self.original_dimensions[0],
            'height': self.original_dimensions[1]
        }
        
        for region in Region:
            rle_data[region.name] = numpy_to_rle(self.masks[region.value])
        
        json.dump(rle_data, open(location, 'w'), indent=4)
        
    def load_masks_rle(self, location):
        rle_data = json.load(open(location, 'r'))
        for region in Region:
            runs = rle_data[region.name]
            self.masks[region.value] = rle_to_numpy(runs, rle_data['width'], rle_data['height'])
    
    def save_masks_numpy(self, location):
        """
        Stacks all the masks into a single nd array (along the first dimension)
        and saves it to the location.
        """
        stacked_masks = np.stack([self.masks[region.value] for region in Region], axis=0)
        np.save(location, stacked_masks)
        
    def set_masks_numpy(self, masks):
        for i, region in enumerate(Region):
            self.masks[region.value] = masks[i]
    
    def load_masks_numpy(self, location):
        stacked_masks = np.load(location)
        for i, region in enumerate(Region):
            self.masks[region.value] = stacked_masks[i]
    
    def load_random_masks(self):
        for region in Region:
            self.masks[region.value] = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
            
def save_image_with_masks(image, masks, location, alpha=0.5):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    overlay = np.zeros_like(image_color, dtype=np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    for region, color in zip(Region, colors):
        colored_mask = np.zeros_like(image_color, dtype=np.uint8)
        for i in range(3):
            colored_mask[:, :, i] = masks[region.value] * color[i]    
        overlay = cv2.add(overlay, colored_mask)
      
    # import pdb; pdb.set_trace()
      
    blended = cv2.addWeighted(image_color, 1.0, overlay, alpha, 0)
    
    cv2.imwrite(location, blended)
            
            
def numpy_to_rle(np_array):
    """
    Note this function does not return width and height of np_array.
    This info is necessary to restore the np_array (via rle_to_numpy).
    """
    runs = []
    flattened = np_array.flatten(order='F') # column-wise flatten, following COCO
    
    run_count = 0
    searching_for = 1 # flips between 0 and 1
    
    # add -1 to force last run to be appended
    for val in list(flattened) + [-1]:
        if val == searching_for or val == -1:
            runs.append(run_count)
            run_count = 1
            searching_for = 1 if searching_for == 0 else 0
        else:
            run_count += 1
    return runs

def rle_to_numpy(runs, width, height):
    bits = []
    which_bit = 0 # starts with bit 0
    for run in runs:
        bits += [which_bit] * run
        which_bit = 1 if which_bit == 0 else 0
    reshaped = np.array(bits).reshape((width, height), order='F')
    return reshaped

