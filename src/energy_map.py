import cv2
import numpy as np
from config import Config
import matplotlib.pyplot as plt
        
class EnergyMap:
    
    @staticmethod
    def compute_energy(image, method=None):
        if method is None:
            method = Config.ENERGY_METHOD
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Compute gradients using Sobel operator at x and y directions
        if method == 'sobel':
            dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            
        energy = np.sqrt(dx**2 + dy**2)# Gradient magnitude:it gives high values at edges
    
        
        if Config.ENERGY_NORMALIZATION:# Normalize energy to [0, 255]
            if energy.max() - energy.min() > 0:
                energy = 255 * (energy - energy.min()) / (energy.max() - energy.min())# Scale to [0,255]
            else:
                energy = np.zeros_like(energy)
        
        return energy.astype(np.float32)
    
    
    @staticmethod
    def visualize_energy(energy_map, title="Carte d'énergie"):# Visualize and save energy map

        plt.figure(figsize=(10, 8))
        plt.imshow(energy_map, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        output_paths = Config.get_output_paths()
        plt.savefig(f"{output_paths['energy_maps']}energy_map.png", dpi=150)
        plt.close()
    
    #compute and save energy map
    @classmethod
    def compute_and_save_energy(cls, image_gray, step=0):
        energy_map = cls.compute_energy(image_gray)# Compute energy map
        #save and visualize energy map at initial and intermediate steps
        if step == 0 or (Config.SAVE_INTERMEDIATE and step % Config.INTERMEDIATE_STEP == 0):
            title = f"Carte d'énergie (étape {step})" if step > 0 else "Carte d'énergie initiale"
            cls.visualize_energy(energy_map, title)
            
            output_paths = Config.get_output_paths()
            np.save(f"{output_paths['energy_maps']}energy_step_{step}.npy", energy_map)
        
        return energy_map