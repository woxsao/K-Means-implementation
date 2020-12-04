import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class KMeansImplementation:
    def __init__(self, k, max_iter, tol):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    def kmean_implement(self, data, blobs = False, categories = None, plus = False):
        if(blobs == True and categories == None):
            print("You must specify a category parameter if blob parameter is True. Please try again")
        else:
            done = False 
            iteration = 0 
            previous_sse = 0    
            
            lowest_sse = None
            lowest_centroid_list = None
            centroid_list = None
            current_sse = None
            #this for loop will generate the best of 10 initial clusters:
            if(plus == False):
                for i in range(0,10):
                    random_centroids = initialize_centroids(data, self.k)

                    assigned_centroids = find_closest_centroid(data, random_centroids)
                    
                    #this section determines the new centroids through averages
                    new_centroids = find_new_centroids(data, random_centroids, assigned_centroids)
                                
                    #this section reassigns the data to the new centroids 
                    new_assigned = find_closest_centroid(data, new_centroids)

                    #this section is for algorithm verification
                    current_sse = get_errors(data, new_assigned, new_centroids)
                    if(lowest_sse == None or current_sse < lowest_sse):
                        lowest_sse = current_sse
                        lowest_centroid_list = random_centroids
                centroid_list = lowest_centroid_list 
            
            else:
                centroid_list = initialize_centroids(data, 1)
                for i in range(0, self.k-1):
                    np.seterr(divide='ignore', invalid='ignore')
                    distances = distance.cdist(data, centroid_list, 'minkowski', p = 1.5)
                    minDistances = np.min(distances, axis = 1)
                    probs = minDistances / sum(minDistances)
                    numPoints = data.shape[0]
                    index = np.random.choice(a = numPoints, p = probs)
                    #point = data[index].reshape(1,len(data.shape))
                    point = data[index].reshape(1, len(data[index]))
                    centroid_list = np.append(centroid_list, point, axis = 0)

            if blobs == True:
                graph_blobs(data, categories, centroid_list, 'initial condition')

            while done == False:
                #this section generates the centroid list
            
                #this section determines which centroid each data point is closest to
                assigned_centroids = find_closest_centroid(data, centroid_list)
                
                
                #this section determines the new centroids through averages
                new_centroids = find_new_centroids(data, centroid_list, assigned_centroids)
                            
                #this section reassigns the data to the new centroids 
                new_assigned = find_closest_centroid(data, new_centroids)

                if blobs == True:
                    graph_blobs(data, categories,new_centroids,iteration)

                #this section is for algorithm verification
                
                current_sse = get_errors(data, new_assigned, new_centroids)
                if (iteration == self.max_iter) or (abs(previous_sse-current_sse) <= self.tol and iteration != 0):
                    done = True
                    return new_assigned, new_centroids, current_sse
                    
                else:
                    previous_sse = current_sse
                    centroid_list = new_centroids
                    iteration += 1
                    done = False
    
def initialize_centroids(points, k):
    shuffled_points = points.copy()
    np.random.shuffle(shuffled_points)
    return shuffled_points[:k]
        
def find_closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = distance.cdist(points, centroids, metric = 'minkowski')
    return_value = np.argmin(distances, axis = 1)
    return return_value
    
def find_new_centroids(points, centroids, assigned_centroids):
    return_array = []
    for i in range(centroids.shape[0]):
        return_array.append(points[assigned_centroids==i].mean(axis = 0)) 
    return np.array(return_array)
    
def get_errors(points, centroids, centroid_coordinates):
    centroids = centroids.astype(int)
    errors = np.sum((points - centroid_coordinates[centroids])**2)
    return errors  
        
def graph_blobs(points, categories, centroid_coordinates, iteration):
    x,y = zip(*centroid_coordinates)
    title = "Iteration #" + str(iteration)
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], marker = 'o', c = categories)
    plt.scatter(x,y, marker = 'h', c = 'red')
    plt.title(title)
    plt.show()
                
    
            
            

