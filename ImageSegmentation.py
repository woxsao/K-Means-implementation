from matplotlib import pyplot as plt
import numpy as np 
from KMeansImplementation import KMeansImplementation
from PIL import Image


def showSegmentationResults(clusterCenters, classification, nbRows, nbCols, k, name, savefig = False):
    """
    This function displays the image
    Parameters:
        - clusterCenters: the centroid coordinates
        - classification: list of indices for each point representing which cluster center it belongs to
        - nbRows: number of rows
        - nbRows: number of columns 
        - k: number of clusters
        - name: string for the filename of the image
        - savefig: if True saves image with name string in jpg format. 
    """
    segmentation = clusterCenters[classification]
    print(segmentation.shape)
    imSeg = segmentation.reshape(nbRows, nbCols, 3).astype(np.uint8)
    plt.figure()
    plt.imshow(Image.fromarray(imSeg))
    
    plt.title('K = '+ str(k))
    plt.draw()
    if(savefig == True):
        plt.savefig(name + str(k) + '.jpg')
    plt.show()
    
def convert_rgb_to_bw(image_array):
    """
    This function converts the rgb image to intensities (BW)
    Parameters:
        - image_array: the array of pixels and RGB values shape of (x,y,3)
    Returns:
        - returns the array of pixels and intensities shape of (x,y,3) 
    """
    bw_array = np.average(image_array, axis = 2)
    return_array = np.repeat(bw_array[:, :, np.newaxis], repeats = 3, axis = 2)
    return return_array

def convert_rgb_to_hsv(r,g,b):
    """
    This function converts an r,g,b coordinate to h,s,v
    Parameters:
        - r: red value
        - g: green value
        - b: blue value
    Returns:
        -h: hue value
        -s: saturation value
        -v: value 
    """
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

def convert_hsv_to_rgb(h,s,v):
    """
    This function converts an h,s,v coordinate to r,g,b. The formulas I used here I found online on Wikipedia
    Parameters:
        - h: hue value
        - s: saturation value
        - v: value
    Returns:
        - r: red value
        - g: green value
        - b: blue value
    """
    r1,g1,b1 = 0,0,0
    c = (v/100)*(s/100)
    x = c*(1-abs(((h/60)%2)-1))
    if(h >=0 and h< 60):
        r1 = c
        g1 = x
        b1 = 0
    elif(h>=60 and h < 120):
        r1 = x
        g1 = c
        b1 = 0
    elif(h >= 120 and h < 180):
        r1 = 0
        g1 = c
        b1 = x
    elif(h >= 180 and h < 240):
        r1 = 0
        g1 = x
        b1 = c
    elif(h >= 240 and h < 300):
        r1 = x
        g1 = 0
        b1 = c
    else:
        r1 = c
        g1 = 0 
        b1 = x
    m = (v/100)-c 
    r = (r1 + m)*255
    g = (g1 + m)*255
    b = (b1 + m) * 255
    return r,g,b

def elbow_plot(k_max, points, k_min = 1):
    """
    This function generates an elbow plot from k_min to k_max
    Parameters:
        - k_max: maximum number of clusters 
        - points: data
        - k_min: the minimum number of clusters, default 1
    """
    x = range(k_min, k_max+1)
    y = []
    for i in range(k_min, k_max+1):
        kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
        sse = kmeans.kmean_implement(points, clustering = 'plus')[2]
        y.append(sse)
    plt.figure()
    plt.plot(x,y, '.b-')
    plt.title('Elbow plot')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()
    
def coordinate_image(image_array, pixel_position = False, weighted = 1):
    """
    This function converts the image array into the proper shape for KMeans evaluation which requires an array
    of shape (n, x) where n is the number of pixels and x is the number of features (3 if RGB, BW intensities, or HSV), 2 for pixel position). 
    Parameters:
        - image_array: numpy array which contains the image data. 
        - if pixel_position is true then we are considering the pixel position for kmeans
        - pixel_position: boolean value representing whether we are considering the pixel positions or not
        - weighted: Only applicable if pixel_position, if True divide pixel positions by 10
    Returns:
        - the flattened array of the image
    """
    pixels = image_array.shape[0] * image_array.shape[1]
    column_counter = 0 
    row_counter = 0
    if pixel_position == True:
        coordinate_array = np.empty((pixels,5))
    else:
        coordinate_array = np.empty((pixels,3)) 
    for i in range(0, pixels):
        for j in range(0,3):
            coordinate_array[i,j] = image_array[row_counter, column_counter,j]
        if pixel_position == True:
            coordinate_array[i,3] = float(row_counter)/weighted
            coordinate_array[i,4] = float(column_counter)/weighted
        if(column_counter == image_array.shape[1]-1):
            column_counter = 0
            row_counter += 1
        else:
            column_counter += 1
    print(coordinate_array.shape)
    return coordinate_array


"""
Tests: 
Sorry for the long commented sections haha
"""
building_image = Image.open("building.jpg")
starfish_image = Image.open("starfish.jpg")
umbrella_image = Image.open("Umbrella.jpg")
clogs_image = Image.open("clogs.jpg")

#indexing 3 columns R G B 
building = (np.asarray(building_image, dtype = np.float32))[:,:,:3]
starfish = (np.asarray(starfish_image, dtype = np.float32))[:,:,:3]
umbrella = (np.asarray(umbrella_image, dtype = np.float32))[:,:,:3]
clogs = (np.asarray(clogs_image, dtype = np.float32))[:,:,:3]

#Vectorizing functions
hsvfunc = np.vectorize(convert_rgb_to_hsv)
rgbfunc = np.vectorize(convert_hsv_to_rgb)


#BW Umbrella Image elbow


"""umbrella_bw = convert_rgb_to_bw(umbrella)
umbrella_flattened = coordinate_image(umbrella_bw)
elbow_plot(10, umbrella_flattened, k_min=1)"""

#BW Umbrella Image Processing

"""for i in range(2, 11):
    umbrella_bw = convert_rgb_to_bw(umbrella)
    umbrella_flattened = coordinate_image(umbrella_bw)
    kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
    assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(umbrella_flattened, clustering = 'plus')

    showSegmentationResults(centroid_coordinates, assigned_centroids, umbrella.shape[0], umbrella.shape[1], i, name = 'BWUmbrella', savefig = True, mode = 'bw')"""

#Elbow plot for umbrella RGB 
"""umbrella_flattened = coordinate_image(umbrella)
elbow_plot(10, umbrella_flattened, k_min=1)"""

"""for i in range(2,11):
    umbrella_flattened = coordinate_image(umbrella)
    kmeans = KMeansImplementation(k = i, max_iter = 200, tol = 0.05)
    assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(umbrella_flattened, clustering = 'plus')

    #my_showSegmentationResults(centroid_coordinates, assigned_centroids, umbrella.shape[0], umbrella.shape[1], i, type = 'bw')
    showSegmentationResults(centroid_coordinates,assigned_centroids,umbrella.shape[0], umbrella.shape[1], i, name = 'RGBUmbrella', savefig = True)"""



#HSV Analysis for Umbrella

"""h,s,v = hsvfunc(umbrella[:,:, 0],umbrella[:,:, 1], umbrella[:,:, 2])
umbrella_hsv = np.dstack((h,s,v))
umbrella_flattened = coordinate_image(np.array(umbrella_hsv))
elbow_plot(10,umbrella_flattened)

for i in range(2, 11):  
    kmeans = KMeansImplementation(k =i, max_iter = 200, tol = 0.05)
    assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(umbrella_flattened, clustering = 'plus')
    r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
    hsv_rgb = np.stack((r,g,b), axis = 1)
    
    print(hsv_rgb.shape)
    print(centroid_coordinates.shape)
    showSegmentationResults(hsv_rgb,assigned_centroids,umbrella.shape[0], umbrella.shape[1], i, name = 'HSVUmbrella' savefig = True)
"""

#Using Pixel Positions Starfish same weight of pixel positions and color channels:
"""starfish_flattened = coordinate_image(starfish, pixel_position=True)
kmeans = KMeansImplementation(k = 5, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(starfish_flattened, clustering = 'plus')
print(centroid_coordinates)
showSegmentationResults(centroid_coordinates[:, :3],assigned_centroids,starfish.shape[0], starfish.shape[1], 5, name = 'StarfishPixelUnWeighted', savefig = True)
"""
#Using Pixel Positions Starfish dividing pixel positions by 3: 
"""starfish_flattened = coordinate_image(starfish, pixel_position=True, weighted= True)
kmeans = KMeansImplementation(k = 5, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(starfish_flattened, clustering = 'plus')
print(centroid_coordinates)
showSegmentationResults(centroid_coordinates[:, :3],assigned_centroids,starfish.shape[0], starfish.shape[1], 5, name = 'StarfishWeightedByThree', savefig = True)"""


#Regular RGB k = 5 for starfish
"""starfish_flattened = coordinate_image(starfish)
kmeans = KMeansImplementation(k = 5, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(starfish_flattened, clustering = 'plus')
showSegmentationResults(centroid_coordinates,assigned_centroids,starfish.shape[0], starfish.shape[1], 5, name = 'StarfishRGB', savefig = True)"""

#Elbow Plot for RGB starfish
"""starfish_flattened = coordinate_image(starfish)
elbow_plot(10,starfish_flattened)"""

#Elbow Plot for HSV starfish
"""h,s,v = hsvfunc(starfish[:,:, 0],starfish[:,:, 1], starfish[:,:, 2])
starfish_hsv = np.dstack((h,s,v))
starfish_flattened = coordinate_image(np.array(starfish_hsv))
elbow_plot(10,starfish_flattened)"""

#k = 4 for HSV starfish (at elbow)
"""h,s,v = hsvfunc(starfish[:,:, 0],starfish[:,:, 1], starfish[:,:, 2])
starfish_hsv = np.dstack((h,s,v))
starfish_flattened = coordinate_image(starfish_hsv)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(starfish_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,starfish.shape[0], starfish.shape[1], 4, name = 'StarfishhSV', savefig = True)
"""
#k = 3 for RGB starfish (at elbow)
"""starfish_flattened = coordinate_image(starfish)
kmeans = KMeansImplementation(k = 3, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(starfish_flattened, clustering = 'plus')
showSegmentationResults(centroid_coordinates,assigned_centroids,starfish.shape[0], starfish.shape[1], 3, name = 'StarfishRGB', savefig = True)"""

#HSV starfish k = 3
"""h,s,v = hsvfunc(starfish[:,:, 0],starfish[:,:, 1], starfish[:,:, 2])
starfish_hsv = np.dstack((h,s,v))
starfish_flattened = coordinate_image(starfish_hsv)
kmeans = KMeansImplementation(k = 3, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(starfish_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,starfish.shape[0], starfish.shape[1], 3, name = 'StarfishhSV', savefig = True)"""

#Testing KMeans vs KMeans++ on Buildings
"""building_flattened = coordinate_image(building)
kmeans = KMeansImplementation(k = 5, max_iter = 200, tol = 0.05)

assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(building_flattened, clustering = 'regular')
showSegmentationResults(centroid_coordinates,assigned_centroids,building.shape[0], building.shape[1], 5, name = 'BuildingRegularKmeans', savefig = True)

assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(building_flattened, clustering = 'random10')
showSegmentationResults(centroid_coordinates,assigned_centroids,building.shape[0], building.shape[1], 5, name = 'BuildingRandom10', savefig = True)

assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(building_flattened, clustering = 'plus')
showSegmentationResults(centroid_coordinates,assigned_centroids,building.shape[0], building.shape[1], 5, name = 'BuildingKMeans++', savefig = True)

"""

#Testing Pixel positions/3 weighted on clogs:
"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(clogs_hsv,pixel_position=True, weighted= 3)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,clogs.shape[0], clogs.shape[1], 4, name = 'ClogsHSVWeighted3', savefig = True)"""

#Testing Pixel positions unweighted on clogs HSV:

"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(clogs_hsv,pixel_position=True)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,clogs.shape[0], clogs.shape[1], 4, name = 'ClogsHSVUnweighted', savefig = True)"""

#Testing Pixel positions/5 weighted on clogs: 
"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(clogs_hsv,pixel_position=True, weighted= 5)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,clogs.shape[0], clogs.shape[1], 4, name = 'ClogsHSVWeighted5', savefig = True)"""

#Testing Pixel positions/2 weighted on clogs: 
"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(clogs_hsv,pixel_position=True, weighted= 2)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,clogs.shape[0], clogs.shape[1], 4, name = 'ClogsHSVWeighted2', savefig = True)"""

#Testing Pixel positions/10 weighted on clogs: 
"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(clogs_hsv,pixel_position=True, weighted= 10)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,clogs.shape[0], clogs.shape[1], 4, name = 'ClogsHSVWeighted11', savefig = True)"""

#Regular RGB For Clogs K = 3
"""clogs_flattened = coordinate_image(clogs)
kmeans = KMeansImplementation(k = 3, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
showSegmentationResults(centroid_coordinates,assigned_centroids,clogs.shape[0], clogs.shape[1], 3, name = 'ClogsRGB', savefig = True)"""

#HSV for Clogs K = 4
"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(clogs_hsv)
kmeans = KMeansImplementation(k = 4, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(clogs_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,clogs.shape[0], clogs.shape[1], 4, name = 'ClogsHSV', savefig = True)"""

#Elbow Plot RGB for Clogs
"""clogs_flattened = coordinate_image(clogs)
elbow_plot(10,clogs_flattened)"""

#Elbow Plot HSV for Clogs
"""h,s,v = hsvfunc(clogs[:,:, 0],clogs[:,:, 1], clogs[:,:, 2])
clogs_hsv = np.dstack((h,s,v))
clogs_flattened = coordinate_image(np.array(clogs_hsv))
elbow_plot(10,clogs_flattened)"""

#elbow plot building image RGB
"""building_flattened = coordinate_image(building)
elbow_plot(10,building_flattened)"""

#elbow plot building image HSV
"""h,s,v = hsvfunc(building[:,:, 0],building[:,:, 1], building[:,:, 2])
building_hsv = np.dstack((h,s,v))
building_flattened = coordinate_image(np.array(building_hsv))
elbow_plot(10,building_flattened)"""

#Building RGB k = 5
"""building_flattened = coordinate_image(building)
kmeans = KMeansImplementation(k = 5, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(building_flattened, clustering = 'plus')
showSegmentationResults(centroid_coordinates,assigned_centroids,building.shape[0], building.shape[1], 5, name = 'BuildingRGB', savefig = True)"""

#Building HSV k = 3
"""h,s,v = hsvfunc(building[:,:, 0],building[:,:, 1], building[:,:, 2])
building_hsv = np.dstack((h,s,v))
building_flattened = coordinate_image(building_hsv)
kmeans = KMeansImplementation(k = 3, max_iter = 200, tol = 0.05)
assigned_centroids, centroid_coordinates, sse = kmeans.kmean_implement(building_flattened, clustering = 'plus')
r,g,b = rgbfunc(centroid_coordinates[:,0], centroid_coordinates[:,1], centroid_coordinates[:,2])
hsv_rgb = np.stack((r,g,b), axis = 1)
showSegmentationResults(hsv_rgb,assigned_centroids,building.shape[0], building.shape[1], 3, name = 'BuildingHSV', savefig = True)"""

