import imgaug.augmenters as iaa
import random
import numpy as np
import cv2

def random_flip(image, masks):
    """ Randomly flip images and masks horizontally and/or vertically """
    if random.random() < 0.5:  # 50% chance
        image = cv2.flip(image, 1)  # Horizontal flip
        masks = [cv2.flip(mask, 1) for mask in masks]
    if random.random() < 0.5:
        image = cv2.flip(image, 0)  # Vertical flip
        masks = [cv2.flip(mask, 0) for mask in masks]
    return image, masks

def random_rotation(image, masks, angle_range=30):
    """ Randomly rotate images and masks within a given angle range """
    angle = random.uniform(-angle_range, angle_range)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    masks = [cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) for mask in masks]
    return image, masks

def random_intensity_shift(image, alpha_range=(0.8, 1.2), beta_range=(-30, 30)):
    """ Randomly adjust brightness and contrast """
    alpha = random.uniform(*alpha_range)  # Contrast factor
    beta = random.uniform(*beta_range)    # Brightness factor
    image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return image

def elastic_deformation(image, masks, alpha=50, sigma=10):
    """ Apply elastic deformations using imgaug """
    aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
    image = aug.augment_image(image)
    masks = [aug.augment_image(mask) for mask in masks]  # Use nearest interpolation for segmentation masks
    return image, masks

def random_crop(image, masks, crop_size=(128, 128)):
    """ Randomly crop image and mask to a fixed size """
    h, w = image.shape[:2]
    ch, cw = crop_size

    if h < ch or w < cw:
        raise ValueError("Crop size must be smaller than image size")

    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)

    image = image[y:y+ch, x:x+cw]
    masks = [mask[y:y+ch, x:x+cw] for mask in masks]
    return image, masks

def add_gaussian_noise(image, mean=0, std=10):
    """ Add Gaussian noise to the image """
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    return np.clip(image, 0, 255)

def random_affine(image, masks, scale_range=(0.9, 1.1), translation_range=10, shear_range=10):
    """ Apply affine transformations: scaling, translation, shearing """
    h, w = image.shape[:2]
    
    # Scaling
    scale = random.uniform(*scale_range)
    M_scale = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)

    # Translation
    tx = random.randint(-translation_range, translation_range)
    ty = random.randint(-translation_range, translation_range)
    M_trans = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    # Shearing
    shear_x = np.tan(np.radians(random.uniform(-shear_range, shear_range)))
    shear_y = np.tan(np.radians(random.uniform(-shear_range, shear_range)))
    M_shear = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)

    # Apply transformations
    image = cv2.warpAffine(image, M_scale, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    image = cv2.warpAffine(image, M_trans, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    image = cv2.warpAffine(image, M_shear, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    masks = [cv2.warpAffine(mask, M_scale, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) for mask in masks]
    masks = [cv2.warpAffine(mask, M_trans, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) for mask in masks]
    masks = [cv2.warpAffine(mask, M_shear, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT) for mask in masks]

    return image, masks

def apply_gaussian_blur(image, kernel_size=3):
    """ Apply Gaussian blur to simulate out-of-focus effect """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def augment_image(image, masks):
    """ Apply a combination of augmentations """
    if random.random() < 0.5:  # 50% chance
        image, masks = random_flip(image, masks)
    if random.random() < 0.5:
        image, masks = random_rotation(image, masks)
    if random.random() < 0.3:
        image = random_intensity_shift(image)
    if random.random() < 0.3:
        image, masks = elastic_deformation(image, masks)
    if random.random() < 0.3:
        image = add_gaussian_noise(image)
    if random.random() < 0.3:
        image, masks = random_affine(image, masks)
    if random.random() < 0.3:
        image = apply_gaussian_blur(image)

    return image, masks