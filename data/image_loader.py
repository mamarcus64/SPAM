import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import multiprocessing

from data.image import Image, Region
from data.image_augmentations import augment_image
from data.training_image import TrainingImage

class TrainingDataset(Dataset):

    def __init__(self, frame_names: list, load_on_init=False, unload_after_access=True, augment=True):
        self.frame_names = frame_names
        self.frames = {}
        self.load_on_init = load_on_init
        self.unload_after_access = unload_after_access
        self.augment = augment

        if self.load_on_init:
            for frame_name in tqdm(self.frame_names, desc="Frames loaded"):
                self.load_frame(frame_name)

    def load_frame(self, frame_name):
        if frame_name not in self.frames: # if already loaded, do nothing
            self.frames[frame_name] = TrainingImage(frame_name)

        return self.frames[frame_name]

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        frame = self.load_frame(self.frame_names[idx])
        
        image = frame.image
        masks = [frame.gold_masks[region.value] for region in Region]
        
        # apply data augmentations
        if self.augment:
            image, masks = augment_image(image, masks)
            
        img_tensor = torch.from_numpy(image).float().unsqueeze(0)
        stacked_masks = np.stack(masks, axis=0)
        mask_tensor = torch.from_numpy(stacked_masks).long()
            
        return img_tensor, mask_tensor

    def __iter__(self):
        self._current_idx = 0  # Initialize the starting index for iteration
        return self

    def __next__(self):
        if self._current_idx >= len(self):
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item


class TrainingLoader(DataLoader):
    """
    A PyTorch DataLoader for sampling frames from the ImageDataset.
    """

    def __init__(
        self,
        frame_names,
        load_on_init=True,
        batch_size=32,
        shuffle=False,
        augment=True,
        num_workers=multiprocessing.cpu_count(),
        unload_after_access=False,
    ):
        self.dataset = TrainingDataset(
            frame_names,
            load_on_init=load_on_init,
            unload_after_access=unload_after_access,
            augment=augment,
        )
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )