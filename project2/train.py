# 1. Find keypoint and extract to feature vector
# 2. Clustering descriptor
# 3. Build BOW
# 4. SVM

from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import time


# 1 Read image

train_path = 'image/train'

classes_name = os.listdir(train_path)


def getFilePaths(folder_path):
    file_paths = []
    for path in os.listdir(folder_path):
        img_path =  os.path.join(folder_path,path)
        file_paths.append(img_path)
    return file_paths

def getFiles(path,classes_name):
    image_paths = []
    image_classes = []
    id = 0
    for class_name in classes_name:
        folder_path = os.path.join(path, class_name)
        class_path = getFilePaths(folder_path)
        image_paths.extend(class_path)
        image_classes.extend([id] * len(class_path))
        id += 1
    return [image_paths,image_classes]



def getDescriptorList(image_paths):
    # List descriptor
    descriptor_list = []
    # sift
    #sift = cv2.SIFT_create()
    
    # brisk
    sift = cv2.BRISK_create()


    # Compute feaature
    t1 = time.time()
    for image_path in image_paths:
        img = cv2.imread(image_path)
        keypoints, descriptor = sift.detectAndCompute(img, None)
        descriptor_list.append(descriptor)
    t2 = time.time()
    print("Feature extraction in %d seconds" %(t2-t1))
    return descriptor_list


def vstackDescriptors(descriptor_list):
    # stack các descriptor lại để phân cụm
    descriptors = descriptor_list[0]
    for descriptor in descriptor_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))
    # kmean works with float
    descriptors_float = descriptors.astype(float)
    return descriptors_float


from scipy.cluster.vq import kmeans, vq
no_clusters = 800
print(no_clusters)
# cluster k , centroid : k center of k clusters
def clusterDescriptor(descriptors,no_clusters):
    t1 = time.time()
    centroids, _ = kmeans(descriptors, no_clusters, 1)
    # centroids, _
    t2 = time.time()
    print("Clustering in %d seconds" %(t2-t1))
    return centroids




def extractFeatures(kmeans,descriptor_list,no_clusters,no_images):
    #no_images = len(image_paths)
    im_features = np.zeros((no_images, no_clusters), "float32")
    # im_features[i][j]: # of j cluster in picture i 
    for i in range(no_images):
        if descriptor_list[i] is not None:
            indexes, distance = vq(descriptor_list[i], kmeans)
           
            for index in indexes:
                im_features[i][index] += 1
    return im_features



from sklearn.preprocessing import StandardScaler
def standardize(im_features):
    scale = StandardScaler().fit(im_features)
    im_std_features = scale.transform(im_features)
    return [scale,im_std_features]


def plotHistogram(vocabulary,no_clusters):
		print( "Plotting histogram")
		x_scalar = np.arange(no_clusters)
		y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(no_clusters)])
  
		print( y_scalar)
		plt.bar(x_scalar, y_scalar)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x_scalar + 0.4, x_scalar)
		plt.show()

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
def svcParamSelection(X, y, nfolds):
		"""	
		param selection 

		"""
		Cs = [0.1, 1, 10, 100, 1000]
		gammas = [1, 0.1, 0.01, 0.001, 0.0001]
		param_grid = {'C': Cs, 'gamma' : gammas}
		grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds,refit = True, verbose = 3, scoring= 'accuracy')
		grid_search.fit(X, y)
		return grid_search


image_paths,image_classes= getFiles(train_path,classes_name)

descriptor_list = getDescriptorList(image_paths)

descriptors = vstackDescriptors(descriptor_list)

# 2. Clusteering
# k centroid
k_centroid = clusterDescriptor(descriptors,no_clusters)

# 3. Build bow
no_images = len(image_paths)
im_features = extractFeatures(k_centroid,descriptor_list,no_clusters,no_images)

# standardlize
stdSlr,im_features = standardize(im_features)

plotHistogram(im_features, no_clusters)
print("Features histogram plotted.")

# 4. Classification
t1 = time.time()
train_labels = np.array(image_classes)
gsSVM= svcParamSelection(im_features,train_labels ,5)
t2 = time.time()
print("Classify in %d seconds" %(t2-t1))


print("Best param")
print(gsSVM.best_params_)
print(gsSVM.best_score_)
print( "Training completed")

# lưu lại mô hình
import pickle
pickle.dump((gsSVM.best_estimator_, classes_name, stdSlr, no_clusters, k_centroid), open('param.pickle', "wb"))

