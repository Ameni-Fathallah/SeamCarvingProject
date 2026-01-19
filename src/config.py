import os

class Config:
    IMAGE_SIZE = (256, 256) 
    MAX_SEAMS_TO_REMOVE = 128# maximum number of seams to remove
    ENERGY_METHOD = 'sobel'#method for energy calculation
    ENERGY_NORMALIZATION = True# whether to normalize energy map
    SAVE_INTERMEDIATE = True
    INTERMEDIATE_STEP = 10# save intermediate results every N seams
    SHOW_VISUALIZATIONS = False
    INPUT_IMAGE_PATH = "images/input/image.jpg"
    OUTPUT_BASE_PATH = "images/output/"
    USE_OPTIMIZED_DAG = True# use optimized DAG construction
    DEBUG_MODE = False
    
    @classmethod
    def get_output_paths(cls):
        return {
            'energy_maps': f"{cls.OUTPUT_BASE_PATH}energy_maps/",
            'seams': f"{cls.OUTPUT_BASE_PATH}seams/",
            'intermediate': f"{cls.OUTPUT_BASE_PATH}intermediate/",
            'final': f"{cls.OUTPUT_BASE_PATH}final/"
        }

output_paths = Config.get_output_paths()
for path in output_paths.values():
    os.makedirs(path, exist_ok=True)