from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np 
from KMeansImplementation import KMeansImplementation

points, categories = make_blobs(n_samples = 150, n_features = 2, centers = 4, cluster_std = 0.5, shuffle = True, random_state = 0)
points = np.array(points)

kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
blob_assignment, centroid_coordinates, sse = kmeans.kmean_implement(points, categories = categories, blobs = True, plus = True)


"""
x = range(1, 10)
y = []
for i in range(1, 10):
    kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
    blob_assignment, centroid_coordinates, sse = kmeans.kmean_implement(points, categories, blobs = True, plus = True)
    y.append(sse)"""
"""plt.figure()
plt.plot(x,y, '.b-')
plt.title('Elbow plot')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()"""