import numpy as np
from collections import defaultdict
from config import Config

class DAGBuilder:
    #this class builds a directed acyclic graph (DAG) from the energy map of the image
    #because seams can only go downwards, the graph is acyclic and directed from top to bottom
    #topological order is simply row by row from top to bottom

    def __init__(self, energy_map):
        self.energy_map = energy_map
        self.height, self.width = energy_map.shape
        self.graph = None
        
        if Config.USE_OPTIMIZED_DAG:
            self.build_optimized_dag()
        else:
            self.build_explicit_dag()
    
    #build explicit DAG representation
    def build_explicit_dag(self):
        self.graph = defaultdict(list)# adjacency list representation
        
        for y in range(self.height - 1):# for each row except the last
            for x in range(self.width):
                current_node = (x, y)# current pixel
                neighbors = []
                
                if x > 0:
                    neighbor = (x - 1, y + 1)# left-down pixel
                    weight = self.energy_map[y + 1, x - 1]# energy of left-down pixel
                    neighbors.append((neighbor, weight))# add to neighbors
                
                neighbor = (x, y + 1)# down pixel
                weight = self.energy_map[y + 1, x]# energy of down pixel
                neighbors.append((neighbor, weight))
                
                if x < self.width - 1:
                    neighbor = (x + 1, y + 1)
                    weight = self.energy_map[y + 1, x + 1]# energy of right-down pixel
                    neighbors.append((neighbor, weight))
                
                self.graph[current_node] = neighbors# set neighbors in graph
    
    #build optimized DAG representation
    def build_optimized_dag(self):
        self.graph = "optimized"
    
    # get neighbors of a node in optimized representation
    def get_neighbors(self, node):
        x, y = node
        
        if y >= self.height - 1:
            return []
        
        neighbors = []
        
        if x > 0:
            neighbor = (x - 1, y + 1)
            weight = self.energy_map[y + 1, x - 1]
            neighbors.append((neighbor, weight))
        
        neighbor = (x, y + 1)
        weight = self.energy_map[y + 1, x]
        neighbors.append((neighbor, weight))
        
        if x < self.width - 1:
            neighbor = (x + 1, y + 1)
            weight = self.energy_map[y + 1, x + 1]
            neighbors.append((neighbor, weight))
        
        return neighbors
    
    # get topological order of nodes : it is simply row by row from top to bottom  return list of nodes in topological order
    def get_topological_order(self):
        order = []
        for y in range(self.height):
            for x in range(self.width):
                order.append((x, y))
        return order