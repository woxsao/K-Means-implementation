"""
TODO:
- coordinate function is currently only tuned for bw images (only have one color value, essentially just traversing
a 2d array instead of a 3d one). Optimize later to support RGB and HSV
"""

from os import DirEntry
from matplotlib import pyplot as plt
from numpy.lib.function_base import vectorize
from sklearn.datasets import make_blobs
import numpy as np 
from KMeansImplementation import KMeansImplementation
from PIL import Image


def showSegmentationResults(clusterCenters, classification, nbRows, nbCols, k, savefig = False, mode = 'rgb'):
    segmentation = clusterCenters[classification]

    imSeg = segmentation.reshape(nbRows, nbCols, 3).astype(np.uint8)
    plt.figure()
    if mode == 'hsv':
        plt.imshow(Image.fromarray(imSeg, mode = 'HSV'))
    else:
        plt.imshow(Image.fromarray(imSeg))
    
    plt.title('K = '+ str(k))
    plt.draw()
    if(savefig == True):
        if mode == 'rgb':
            plt.savefig('RGBK' + str(k) + 'Umbrella.jpg')
        elif mode == 'bw':
            plt.savefig('BWK' + str(k) + 'Umbrella.jpg')
        else:
            plt.savefig('HSVK' + str(k) + "Umbrella.jpg")
    plt.show()
    
def convert_rgb_to_bw(image_array):
    print('original image array: ', image_array.shape)
    bw_array = np.average(image_array, axis = 2)
    return_array = np.repeat(bw_array[:, :, np.newaxis], repeats = 3, axis = 2)
    print('bw array shape: ', return_array.shape)
    return return_array

def convert_rgb_to_hsv(r,g,b):
    r = r/255.0
    g = g/255.0
    b = b/255.0
    color_max = max(r,g,b)
    color_min = min(r,g,b)
    color_diff = color_max-color_min
    if color_max == color_min:
        h = 0
    elif color_max == r:
        h = 60*(((g-b)/color_diff) % 6)
    elif color_max == g:
        h = 60*(((b-r)/color_diff) +2)
    else:
        h = 60*(((r-g)/color_diff)+4)
    if color_max == 0:
        s = 0
    else:
        s = (color_diff/color_max)*100
    v = color_max *100
    return h,s,v

def dup_columns(a, index, num = 1):
    return np.insert(a, [index+1]*num,a[:, index], axis = 2)

def elbow_plot(k_max, points, k_min = 1):
    x = range(k_min, k_max+1)
    y = []
    for i in range(k_min, k_max+1):
        kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
        blob_assignment, centroid_coordinates, sse = kmeans.kmean_implement(points, plus = True)
        y.append(sse)
    plt.figure()
    plt.plot(x,y, '.b-')
    plt.title('Elbow plot')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
    
def coordinate_image(image_array):
    """
    This function converts the image array into the proper shape for KMeans evaluation which requires an array
    of shape (n, 2) where n is the number of pixels. 
    image_array: numpy array which contains the image data. 
    """
    pixels = image_array.shape[0] * image_array.shape[1]
    column_counter = 0 
    row_counter = 0

    coordinate_array = np.empty((pixels,3)) 
    dimension = 3     
    for i in range(0, pixels):
        if dimension == 3:
            for j in range(0,dimension):
                coordinate_array[i,j] = image_array[row_counter, column_counter,j]
        elif dimension == 2:
            for j in range(0, dimension):
                coordinate_array[i,j] = image_array[row_counter, column_counter]
        if(column_counter == image_array.shape[1]-1):
            column_counter = 0
            row_counter += 1
        else:
            column_counter += 1
    """shuffled_array = coordinate_array.copy()
    np.random.shuffle(shuffled_array)
    print(shuffled_array[0:100,:])"""
    return coordinate_array


"""
Main code:
"""
hanbok_image = Image.open("hanbok.jpg")
starfish_image = Image.open("starfish.jpg")
umbrella_image = Image.open("Umbrella.jpg")
#indexing 3 columns R G B 
hanbok = (np.asarray(hanbok_image, dtype = np.float32))[:,:,:3]
starfish = (np.asarray(starfish_image, dtype = np.float32))[:,:,:3]
umbrella = (np.asarray(umbrella_image, dtype = np.float32))[:,:,:3]



#Umbrella Image Processing

"""
print("sailboats_bw shape: ", umbrella_bw.shape)
umbrella_flattened = coordinate_image_2d(umbrella_bw)
elbow_plot(10, umbrella_flattened, k_min=1)"""

#BW Umbrella Image Processing

"""for i in range(2, 11):
    umbrella_bw = convert_rgb_to_bw(umbrella)
    umbrella_flattened = coordinate_image(umbrella_bw)
    kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
    assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(umbrella_flattened, plus = True)

    showSegmentationResults(centroid_coordinates, assigned_centroids, umbrella.shape[0], umbrella.shape[1], i,savefig = True, mode = 'bw')"""


"""umbrella_flattened = coordinate_image(umbrella)
elbow_plot(10, umbrella_flattened, k_min=2)"""

"""for i in range(2,11):
    umbrella_flattened = coordinate_image(umbrella)
    kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
    assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(umbrella_flattened, plus = True)

    #my_showSegmentationResults(centroid_coordinates, assigned_centroids, umbrella.shape[0], umbrella.shape[1], i, type = 'bw')
    showSegmentationResults(centroid_coordinates,assigned_centroids,umbrella.shape[0], umbrella.shape[1], i, savefig = True)"""

#convert RGB to HSV

print(umbrella[:, :, 0].shape)
hsvfunc = np.vectorize(convert_rgb_to_hsv)

for i in range(2, 11):
    h,s,v = hsvfunc(umbrella[:,:, 0],umbrella[:,:, 1], umbrella[:,:, 2])
    umbrella_hsv = np.dstack((h,s,v))
    umbrella_flattened = coordinate_image(np.array(umbrella_hsv))
    kmeans = KMeansImplementation(k =i, max_iter = 200, tol = 0.05)
    assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(umbrella_flattened, plus = True)
    print(centroid_coordinates)
    showSegmentationResults(centroid_coordinates,assigned_centroids,umbrella.shape[0], umbrella.shape[1], i, savefig = True, mode = 'hsv')
