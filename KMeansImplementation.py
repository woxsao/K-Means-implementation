import numpy as np
class K_Means:
    def __init__(self, k, tol = 0.001, max_iter = 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def centroids(self, data):

        #this section generates the centroid list
        centroid_list = []
        for i in range(self.k):
            centroid_list[i] = np.random.choice(data)

        #this section determines which centroid each data point is closest to
        assigned_centroids = []
        
        for i in data:
            closest = centroid_list[0]
            closest_distance = np.linalg.norm(centroid_list[0]-i)
            for j in centroid_list:
                dist = np.linalg.norm(j-i)
                if dist < closest_distance:
                    closest_distance = dist
                    closest = j 
            np.append(assigned_centroids, closest, axis = 0)

        classification = np.append(data,assigned_centroids, axis = 1) 
        
        #this section determines the new centroids
        new_centroids = []
        for i in centroid_list:
            temp = []
            for j in classification:
                if j[1] == i:
                    np.append(temp, j[0])
            np.append(new_centroids, np.average(temp))
        
        #this section reassigns the data to the new centroids 
        new_assigned = []
        for i in data:
            closest = new_centroids[0]
            closest_distance = np.linalg.norm(new_centroids[0]-i)
            for j in new_centroids:
                dist = np.linalg.norm(j-i)
                if dist < closest_distance:
                    closest_distance = dist
                    closest = j 
            np.append(new_assigned, closest, axis = 0)        