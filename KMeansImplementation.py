"""
TODO:
- change distance to scipy minkowski distance
- insert blob graphs 

"""

import numpy as np
import matplotlib as plt

class KMeansImplementation:
    def __init__(self, k, max_iter, tol):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def kmean_implement(self, data):
        done = False 
        iteration = 0 
        previous_sse = 0    
        
        ten_centroids = np.array(10,1)
        ten_sse = np.array(10,1)
        #this for loop will generate the best of 10 initial clusters:

        for i in range(0,10):
            centroid_list = initialize_centroids(data, self.k)

            assigned_centroids = find_closest_centroid(data, centroid_list)
            
            #this section determines the new centroids through averages
            new_centroids = find_new_centroids(data, centroid_list, assigned_centroids)
                        
            #this section reassigns the data to the new centroids 
            new_assigned = find_closest_centroid(data, new_centroids)

            #this section is for algorithm verification
            sse = get_errors(data, new_assigned, new_centroids)
            ten_centroids[i] = new_centroids
            ten_sse[i] = sse

        centroid_list = ten_centroids[np.argmin(ten_sse)]
        while done == False:
            #this section generates the centroid list
        
            #this section determines which centroid each data point is closest to
            assigned_centroids = find_closest_centroid(data, centroid_list)
            
            #this section determines the new centroids through averages
            new_centroids = find_new_centroids(data, centroid_list, assigned_centroids)
                        
            #this section reassigns the data to the new centroids 
            new_assigned = find_closest_centroid(data, new_centroids)

            #this section is for algorithm verification
            
            current_sse = get_errors(data, new_assigned, new_centroids)
            if (iteration == self.max_iter) or (abs(previous_sse-current_sse) <= self.tol and iteration != 0):
                done = True
                print("Previous SSE: ", previous_sse)
                print( "Current SSE: ", current_sse)
                print("difference: ", abs(current_sse-previous_sse))
                print(iteration)
                return np.concatenate((data, new_assigned[:,np.newaxis]), axis = 1), new_centroids
                
            else:
                print("Previous SSE: ", previous_sse)
                print( "Current SSE: ", current_sse)
                print("difference: ", abs(current_sse-previous_sse))
                previous_sse = current_sse
                centroid_list = new_centroids
                iteration += 1
                print(iteration)
                done = False
    
def initialize_centroids(points, k):
    shuffled_points = points.copy()
    np.random.shuffle(shuffled_points)
    return shuffled_points[:k]
        
def find_closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.linalg.norm(centroids[:, np.newaxis] - points, axis = 2)
    return np.argmin(distances, axis=0)
    
def find_new_centroids(points, centroids, assigned_centroids):
    return_array = []
    for i in range(centroids.shape[0]):
        return_array.append(points[assigned_centroids==i].mean(axis = 0)) 
    return np.array(return_array)
    
def get_errors(points, centroids, centroid_coordinates):
    centroids = centroids.astype(int)
    errors = np.sum((points - centroid_coordinates[centroids])**2)
    return errors  
        
    
                
    
            
            

