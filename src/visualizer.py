import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class Visualizer:
    #this class handles visualization and saving of images during seam carving process
    
    # static method to draw seam on image
    @staticmethod
    def draw_seam_on_image(image, seam_indices, thickness=1):
        image_with_seam = image.copy()
        height = image.shape[0]
        
        for y in range(height):
            x = seam_indices[y]
            if 0 <= x < image.shape[1]:
                cv2.circle(image_with_seam, (x, y), thickness, (0, 0, 0), -1)
        
        return image_with_seam
    

    #static method to visualize each step : image with seam, energy map with seam, reduced image    
    @staticmethod
    def visualize_step(image_rgb, seam_indices, energy_map, step, total_steps):
        if not Config.SAVE_INTERMEDIATE:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        image_with_seam = Visualizer.draw_seam_on_image(image_rgb, seam_indices)
        axes[0].imshow(image_with_seam)
        axes[0].set_title(f'Image avec seam (étape {step+1}/{total_steps})')
        axes[0].axis('off')
        
        energy_display = energy_map.copy()
        if energy_display.max() > 0:
            energy_display = (energy_display / energy_display.max() * 255)
        energy_display = energy_display.astype(np.uint8)
        
        energy_with_seam = Visualizer.draw_seam_on_image(energy_display, seam_indices)
        axes[1].imshow(energy_with_seam, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Carte d\'énergie avec seam')
        axes[1].axis('off')
        
        axes[2].imshow(image_rgb)
        axes[2].set_title(f'Image réduite ({image_rgb.shape[1]}px)')
        axes[2].axis('off')
        
        plt.suptitle(f'Seam Carving - Étape {step+1}/{total_steps}', fontsize=16)
        plt.tight_layout()
        
        if Config.SAVE_INTERMEDIATE and (step == 0 or (step + 1) % Config.INTERMEDIATE_STEP == 0):
            output_paths = Config.get_output_paths()
            plt.savefig(f"{output_paths['intermediate']}step_{step+1:03d}.png", dpi=150)
        
        plt.close()
    

    #static method to create comparison image
    @staticmethod
    def create_comparison(original_image, final_image):
        if original_image.shape[0] != final_image.shape[0]:
            final_image = cv2.resize(
                final_image, 
                (final_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        separator = np.full((original_image.shape[0], 5, 3), 255, dtype=np.uint8)
        comparison = np.hstack([original_image, separator, final_image])
        
        return comparison
    
    #static method to save final results
    @staticmethod
    def save_final_results(original_image, final_image, original_width, final_width):
        output_paths = Config.get_output_paths()
        
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_paths['final']}final_image.png", final_image_bgr)
        
        comparison = Visualizer.create_comparison(original_image, final_image)
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_paths['final']}comparison.png", comparison_bgr)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_image)
        axes[0].set_title(f'Originale: {original_width}px')
        axes[0].axis('off')
        
        axes[1].imshow(final_image)
        axes[1].set_title(f'Résultat: {final_width}px')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_paths['final']}final_comparison_plot.png", dpi=150)
        plt.close()