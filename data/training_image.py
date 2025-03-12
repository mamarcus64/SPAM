import numpy as np
import pdb
import cv2
import random

from data.image import Image, Region


class TrainingImage(Image):
    """
    Represents a single frame of a video, loaded from an `.npz` file.
    Special data format derived from training.
    """

    def __init__(self, frame_path: str):
        # bad practice, but just recreating the __init__ of Image with extra steps
        # instead of calling super().__init__ (wouldn't work properly)
        with np.load(frame_path) as data:
            
            self.original_image = cv2.cvtColor(data['image'], cv2.COLOR_BGR2GRAY)
            self.original_dimensions = self.original_image.shape
            
            if self.original_dimensions == (Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT):
                self.image = self.original_image
            else:
                self.image = cv2.resize(self.original_image, (Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT), interpolation=cv2.INTER_LINEAR)
       
            # self.image = cv2.fastNlMeansDenoising(
                # self.original_image, None, h=11, templateWindowSize=7, searchWindowSize=21
            # )
            
            # create masks
            self.masks = {}
            for region in Region:
                # note to self: make sure to stay consistent with np.uint8!
                self.masks[region.value] = np.zeros(shape=(Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT), dtype=np.uint8)
                
            # load gold masks
            self.gold_masks = {}
            # hard coding bc lazy
            self.gold_masks[0] = self.get_mask_contour(data['jaw_contour'])
            self.gold_masks[1] = self.get_mask_contour(data['nasal_contour'])
            self.gold_masks[2] = self.get_mask_contour(data['neck_contour'])

    def get_mask_contour(self, contour_points) -> np.ndarray:

        frame_shape = self.image.shape[0:2]

        # Create an empty binary mask
        mask = np.zeros(frame_shape, dtype=np.uint8)

        # Convert the list of points into a NumPy array of shape (N, 1, 2)
        polygon = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))

        # Fill the polygon on the mask with 1s
        cv2.fillPoly(mask, [polygon], 1)

        return mask