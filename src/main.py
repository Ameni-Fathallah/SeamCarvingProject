import sys
import os
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_loader import ImageLoader
from src.energy_map import EnergyMap
from src.dag_builder import DAGBuilder
from src.shortest_path_dag import ShortestPathDAG
from src.seam_removal import SeamRemover
from src.visualizer import Visualizer
from src.utils import Timer, print_progress
from src.config import Config

def main():
    print("SEAM CARVING - Redimensionnement intelligent")
    print("Mode : Exécution automatique")
    
    try:
        # step 1: Load and prepare the image
        with Timer("Chargement de l'image"):
            image_rgb, image_gray = ImageLoader.load_and_prepare_image()
        
        original_image = image_rgb.copy()
        original_width = image_rgb.shape[1]
        
        # determine how many seams to remove (limited by config and half the width)
        K = min(Config.MAX_SEAMS_TO_REMOVE, original_width // 2)
        

        # Step 2: Compute the initial energy map
        with Timer("Calcul de la carte d'énergie initiale"):
            energy_map = EnergyMap.compute_and_save_energy(image_gray, step=0)
        
        #step 3: Iteratively remove seams
        for step in range(K):
            print_progress(step, K, prefix="Progression:", suffix="Complet")
            
            #build the DAG (Directed Acyclic Graph) from the energy map
            with Timer("Construction du DAG"):
                dag = DAGBuilder(energy_map)
            
            with Timer("Recherche du seam optimal"):
                shortest_path_finder = ShortestPathDAG(dag)
                seam_indices, seam_energy = shortest_path_finder.find_optimal_seam()
            
            Visualizer.visualize_step(image_rgb, seam_indices, energy_map, step, K)
            
            with Timer("Suppression du seam"):
                image_rgb, image_gray, energy_map = SeamRemover.process_seam_removal(
                    image_rgb, image_gray, energy_map, seam_indices, step
                )
            
            with Timer("Recalcul de l'énergie"):
                energy_map = EnergyMap.compute_energy(image_gray)
        
        # step 4: Summarize results
        final_width = image_rgb.shape[1]
        seams_removed = original_width - final_width
        
        print(f"\nRÉSUMÉ DU TRAITEMENT:")
        print(f"Largeur originale : {original_width} px")
        print(f"Largeur finale    : {final_width} px")
        print(f"Seams supprimés   : {seams_removed}")
        print(f"Réduction         : {(seams_removed/original_width*100):.1f}%")
        
        # step 4: dave final results
        with Timer("Sauvegarde des résultats"):
            Visualizer.save_final_results(original_image, image_rgb, original_width, final_width)
        
        output_paths = Config.get_output_paths()
        print(f"\nRÉSULTATS DISPONIBLES DANS :")
        print(f"Étapes intermédiaires : {output_paths['intermediate']}")
        print(f"Résultats finaux : {output_paths['final']}")
        print(f"Image finale : {output_paths['final']}final_image.png")
        
    except FileNotFoundError as e:
        print(f"ERREUR : {e}")
        print("Solution : Placez une image dans 'images/input/' nommée 'image.jpg'")
        return 1
    except Exception as e:
        print(f"ERREUR : {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())