import numpy as np
from config import Config

class ShortestPathDAG:
    #this class finds the shortest path (minimum energy seam) in the DAG representation of the image
    #with a dynamic programming approach that leverages the topological order of the DAG 
    def __init__(self, dag_builder):
        self.dag = dag_builder
        self.height = dag_builder.height
        self.width = dag_builder.width
        self.dist = np.full((self.height, self.width), np.inf, dtype=np.float32)# distance array
        self.prev = np.full((self.height, self.width, 2), -1, dtype=np.int32)# previous node array
    
    def find_optimal_seam(self):
        self.dist[0, :] = self.dag.energy_map[0, :]# initialize top row distances : energy values of top row pixels
        # relax edges in topological order : for each pixel, update distances of its neighbors 
        for y in range(self.height - 1):
            for x in range(self.width):
                current_dist = self.dist[y, x]
                
                if current_dist == np.inf:
                    continue
                
                neighbors = []
                
                if x > 0:
                    neighbors.append((x - 1, y + 1))
                
                neighbors.append((x, y + 1))
                
                if x < self.width - 1:
                    neighbors.append((x + 1, y + 1))
                
                for nx, ny in neighbors:# for each neighbor of current pixel new_dist = current_dist + weight
                    new_dist = current_dist + self.dag.energy_map[ny, nx]
                    
                    #if new distance is better, update distance and previous node
                    if new_dist < self.dist[ny, nx]:
                        self.dist[ny, nx] = new_dist
                        self.prev[ny, nx] = [x, y]
        
        # find the end of the optimal seam by looking for the minimum distance in the last row
        last_row = self.dist[self.height - 1, :]
        end_x = np.argmin(last_row)# index of minimum distance in last row
        total_energy = last_row[end_x]
        
        seam_indices = self._backtrack_seam(end_x)# reconstruct seam from end to start
        
        return seam_indices, total_energy
    
    #a method to backtrack the seam from the end position
    def _backtrack_seam(self, end_x):
        seam = []
        x, y = end_x, self.height - 1
        
        seam.append(x)
        
        while y > 0:
            prev_x, prev_y = self.prev[y, x]
            seam.append(prev_x)
            x, y = prev_x, prev_y
        
        seam = seam[::-1]
        
        for y in range(self.height):
            seam_x = seam[y]
            if seam_x < 0 or seam_x >= self.width:
                raise ValueError(f"Seam incohérent à la ligne {y}: x={seam_x}")
        
        return seam