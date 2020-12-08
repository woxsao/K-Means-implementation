import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class KMeansImplementation:
    def __init__(self, k, max_iter, tol):
        """
        Constructor: k is the number of clusters
                     max_iter is the maximum number of iterations
                     tol is the tolerance margin for the SSE
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def kmean_implement(self, points, blobs = False, categories = None, clustering = 'regular'):
        """
        This function uses kmeans clustering to segment the points.
        Parameters:
            - points is the points that we are fitting to the algorithm
            - blobs is a boolean for plotting blobs
            - categories is a paramter for the blobs graphing
            - clustering is the intial condition centroid choosing method. This function supports random initial cluster choice, best of 10
              which is just choosing the centroids that produces the lowest SSE out of 10 random centroid combinations, and also kmeans++ 
        Returns:
            - a list of indeces representing which cluster each point belongs to
            - the coordinates of each of those cluster centers
            - the final Sum of square errors (SSE)
        """
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
            if(clustering == 'random10'):
                for i in range(0,10):
                    random_centroids = initialize_centroids(points, self.k)
                    current_sse, new_centroids = kmeans_skeleton(points, random_centroids)[0:2]
                    if(lowest_sse == None or current_sse < lowest_sse):
                        lowest_sse = current_sse
                        lowest_centroid_list = new_centroids
                centroid_list = lowest_centroid_list 

            #This is the implementation for kmeans++
            elif clustering == 'plus':
                centroid_list = initialize_centroids(points, 1)
                for i in range(0, self.k-1):
                    #np.seterr(divide='ignore', invalid='ignore')
                    distances = distance.cdist(points, centroid_list, 'minkowski', p = 1.5)
                    minDistances = np.min(distances, axis = 1)
                    probs = minDistances / sum(minDistances)
                    numPoints = points.shape[0]
                    index = np.random.choice(a = numPoints, p = probs)
                    point = points[index].reshape(1, len(points[index]))
                    centroid_list = np.append(centroid_list, point, axis = 0)

            #This is random centroid initialization, no bells or whistles
            else:
                centroid_list = initialize_centroids(points, self.k)
            
            #code for blob graphing
            if blobs == True:
                graph_blobs(points, categories, centroid_list, 'initial condition')

            while done == False:
                current_sse, new_centroids, new_assigned = kmeans_skeleton(points, centroid_list)
                #code for blob graphing
                if blobs == True:
                    graph_blobs(points, categories,new_centroids,iteration)
                #this section is to check whether we need to keep iterating. If the SSE hasn't changed less than the tol, then we're done
                if (iteration == self.max_iter) or (abs(previous_sse-current_sse) <= self.tol and iteration != 0):
                    done = True
                    print(iteration)
                    return new_assigned, new_centroids, current_sse
                    
                else:
                    previous_sse = current_sse
                    centroid_list = new_centroids
                    iteration += 1
                    done = False

def kmeans_skeleton(points, centroid_list):
    """
    This function runs the kmeans algorithm for one iteration, just helps shorten my code a little 
    Parameters:
        - points: the data
        - centroid_list: the list of centroid coordinates
    Returns:
        - The sum of square errors 
        - The centroid list
        - the list of indices indicating which centroid each point is clustered to
    """
    assigned_centroids = find_closest_centroid(points, centroid_list)
                    
    #this section determines the new centroids through averages
    new_centroids = find_new_centroids(points, centroid_list, assigned_centroids)
                                
    #this section reassigns the points to the new centroids 
    new_assigned = find_closest_centroid(points, new_centroids)

    #this section is for algorithm verification
    current_sse = get_errors(points, new_assigned, new_centroids)        
    return current_sse, new_centroids, new_assigned

def initialize_centroids(points, k):
    """
    This function generates k random centroid points from the points
    Parameters:
        - points is the points
        - k is the number of centroids
    Returns:
        - an array of k centroids
    """
    shuffled_points = points.copy()
    np.random.shuffle(shuffled_points)
    return shuffled_points[:k, :]
        
def find_closest_centroid(points, centroids):
    """
    returns an array containing the index to the nearest centroid for each point
    Parameters:
        - points: the points 
        - centroids: the coordinates of the centroids
    Returns:
        - an array of which indices representing which centroid each point is closest to. 
    """
    distances = distance.cdist(points, centroids, metric = 'minkowski')
    return_value = np.argmin(distances, axis = 1)
    return return_value
    
def find_new_centroids(points, centroids, assigned_centroids):
    """
    returns a list of new centroid coordinates based on the averages across each cluster
    Parameters:
        - points: the data
        - centroids: the coordinates of the centroids
    Returns:
        - an array of new averaged centroid coordinates
    """
    return_array = []
    for i in range(centroids.shape[0]):
        return_array.append(points[assigned_centroids==i].mean(axis = 0)) 
    return np.array(return_array)
    
def get_errors(points, assigned_centroids, centroid_coordinates):
    """
    returns the sum of squared errors between centroids and the data. 
    Parameters: 
        - points: data 
        - assigned_centroids: list of indices representing which centroid each data point is assigned to
        - centroid_coordinates: list of each centroid's coordinates
    Returns:
        - a float representing the sum of squared errors
    """
    centroids = assigned_centroids.astype(int)
    errors = np.sum((points - centroid_coordinates[centroids])**2)
    return errors  
        
def graph_blobs(points, categories, centroid_coordinates, iteration):
    """
    Graphs the blob and iteration, only relevant for the blobs from sklearn
    Parameters:
        - poitns: data
        - categories: the original categories from the sklearn blob function
        - centroid_coordinates: the coordinates for the centroids
        - iteration: which iteration number the algorithm is on
    """
    x,y = zip(*centroid_coordinates)
    title = "Iteration #" + str(iteration)
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], marker = 'o', c = categories)
    plt.scatter(x,y, marker = 'h', c = 'red')
    plt.title(title)
    plt.show()
                

            
            

