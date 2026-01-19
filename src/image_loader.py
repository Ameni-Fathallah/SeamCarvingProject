import cv2# OpenCV for image processing
import numpy as np
from config import Config

class ImageLoader:
    
    #load image from path
    @staticmethod
    def load_image(image_path=None):
        if image_path is None:
            image_path = Config.INPUT_IMAGE_PATH
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    #resize image to target size
    @staticmethod
    def resize_image(image, target_size=None):
        if target_size is None:
            target_size = Config.IMAGE_SIZE
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def to_grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    #normalize image to uint8
    @staticmethod
    def normalize_image(image):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    #validate image size
    @staticmethod
    def validate_image(image):
        if image.shape[:2] != Config.IMAGE_SIZE[::-1]:
            raise ValueError(
                f"L'image doit avoir la taille {Config.IMAGE_SIZE}. "
                f"Taille actuelle : {image.shape[1]}x{image.shape[0]}"
            )
        return True
    
    #load and prepare image
    @classmethod
    def load_and_prepare_image(cls, image_path=None):
        image_rgb = cls.load_image(image_path)# Load image BGR(openCV ) and convert to RGB
        
        if image_rgb.shape[:2] != Config.IMAGE_SIZE[::-1]:# Resize if not matching target size
            image_rgb = cls.resize_image(image_rgb)
        
        cls.validate_image(image_rgb)
        # Convert to grayscale
        image_gray = cls.to_grayscale(image_rgb)
        image_rgb = cls.normalize_image(image_rgb)
        image_gray = cls.normalize_image(image_gray)
        
        return image_rgb, image_gray