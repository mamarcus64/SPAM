import torch
import numpy as np

from src.unet_model import UNet
from data.image import Image, Region

# singleton class that loads U-Net model
class PointExtractor:
    _instance = None
    _model = None
    _device = None
    
    def __new__(cls, model_path=None, device='cuda'):
        if cls._instance is None: # create singleton
            cls._device = device
            cls._instance = super(PointExtractor, cls).__new__(cls)
            
            # load weights into model
            cls._model = UNet()
            cls._model.load_state_dict(torch.load(model_path, map_location=device))
            cls._model = cls._model.to(device)
            cls._model.eval()
        return cls._instance
    
    def extract_points(self, image):
        while len(image.shape) < 4: # add dimensions
            image = image[np.newaxis, :]

        assert image.shape == (1, 1, Image.RESIZED_WIDTH, Image.RESIZED_HEIGHT)
        
        image = torch.from_numpy(image).float()
        
        image = image.to(self._device)
        
        points = []
        
        with torch.no_grad():
            output = self._model(image)
            output = torch.sigmoid(output)
            
            prediction = (output >= 0.5).to(torch.uint8)
            
            for region in Region:
                points.append(get_center_of_mass(prediction[0, region.value]))
        
        return points
                
def get_center_of_mass(segmentation_mask):
    """Compute the center-of-mass for each predicted segmentation."""
    if torch.sum(segmentation_mask) == 0:
        return None  # No valid region found
    
    indices = torch.nonzero(segmentation_mask, as_tuple=False).float()
    
    centroid = indices.mean(dim=0)
    
    # import pdb; pdb.set_trace()
    
    return tuple([int(x) for x in centroid.tolist()])