from sklearn.datasets import make_blobs
import numpy as np 
from KMeansImplementation import KMeansImplementation

points, categories = make_blobs(n_samples = 150, n_features = 2, centers = 4, cluster_std = 0.5, shuffle = True, random_state = 0)
points = np.array(points)

kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
blob_assignment, centroid_coordinates = kmeans.kmean_implement(points)
print(blob_assignment)
print(centroid_coordinates)

"""def initialize_centroids(points, k):
    shuffled_points = points.copy()
    np.random.shuffle(shuffled_points)
    return shuffled_points[:k]

centroid_list = np.array(initialize_centroids(points, 4))

def find_closest_centroid(points, centroids):
    #returns an array containing the index to the nearest centroid for each point
    distances = np.linalg.norm(centroids[:, np.newaxis] - points, axis = 2)
    return np.argmin(distances, axis=0)

closest_centroids = find_closest_centroid(points, centroid_list)


def find_new_centroids(points, centroids, assigned_centroids):
        return_array = []
        for i in range(centroids.shape[0]):
            return_array.append(points[assigned_centroids==i].mean(axis = 0)) 
        return np.array(return_array)
    

print(closest_centroids)
print(find_new_centroids(points, centroid_list, closest_centroids))"""