import numpy as np
import cv2
from config import Config

class SeamRemover:
    #this class handles the removal of seams from images and energy maps it reduces the width of the image by one pixel along the specified seam
    
    @staticmethod
    def remove_seam(image, seam_indices, is_color=True):
        height, width = image.shape[:2]
        
        if len(seam_indices) != height:
            raise ValueError(f"Le seam doit avoir {height} indices, a {len(seam_indices)}")
        
        #case 1 : color image(3 channels)
        if is_color and len(image.shape) == 3:
            new_width = width - 1
            #create new image with one less column
            new_image = np.zeros((height, new_width, 3), dtype=image.dtype)
            
            #for each row , remove the pixel at seam index
            for y in range(height):
                seam_x = seam_indices[y]
                new_image[y, :seam_x, :] = image[y, :seam_x, :]# copy left part of the row to seam
                new_image[y, seam_x:, :] = image[y, seam_x+1:, :]# copy right part after seam
        
        #case 2 : grayscale image(single channel)
        else:
            
            new_width = width - 1
            #crzate a new grayscale image with one less column
            new_image = np.zeros((height, new_width), dtype=image.dtype)
            
            for y in range(height):
                seam_x = seam_indices[y]
                new_image[y, :seam_x] = image[y, :seam_x]
                new_image[y, seam_x:] = image[y, seam_x+1:]
        
        return new_image# return the image with seam removed
    
    @staticmethod
    def remove_seam_from_energy(energy_map, seam_indices):
        #remove seam from energy map (grayscale)
        return SeamRemover.remove_seam(energy_map, seam_indices, is_color=False)
    
    @classmethod
    def process_seam_removal(cls, image_rgb, image_gray, energy_map, seam_indices, step):
        new_image_rgb = cls.remove_seam(image_rgb, seam_indices, is_color=True)# remove seam from the RGB image
        new_image_gray = cls.remove_seam(image_gray, seam_indices, is_color=False)# remove seam from the grayscale image
        new_energy_map = cls.remove_seam_from_energy(energy_map, seam_indices)# remove seam from the energy map
        
        new_height, new_width = new_image_rgb.shape[:2]
        
        #return all updated versions : RGB image, grayscale image, energy map
        return new_image_rgb, new_image_gray, new_energy_map