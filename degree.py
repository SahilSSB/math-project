import numpy as np
import adjacency as a
 
 
def degree(adjacencyMatrix):
    degrees = np.sum(adjacencyMatrix, axis=1)      
    degreeMatrix = np.diag(degrees)                
    return degrees, degreeMatrix                  
 
