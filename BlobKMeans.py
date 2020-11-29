from sklearn.datasets import make_blobs
import numpy as np 

points, categories = make_blobs(n_samples = 150, n_features = 2, centers = 4, cluster_std = 0.5, shuffle = True, random_state = 0)
