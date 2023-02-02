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
cv2.__version__

# 1 Đọc. ảnh vào và tìm các descriptor
# path tới folder ảnh train
train_path = 'image/train'
# list các classes
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
    # list chứa các descriptor của ảnh
    descriptor_list = []
    # sử dụng sift với 128 feature cho mỗi keypoint phát hiện trong ảnh
    sift = cv2.SIFT_create()
     # dung surf
    #sift = cv2.BRISK_create(30)


    # Bước này đọc các ảnh và áp dụng sift lên ảnh
    t1 = time.time()
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # im = cv2.resize(im, (150,150))
        keypoints, descriptor = sift.detectAndCompute(img, None)
        descriptor_list.append(descriptor)
    t2 = time.time()
    print("Done feature extraction in %d seconds" %(t2-t1))
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
no_clusters = 500
# phân thành 500 cụm, giá trị voc trả về là 500 center của 500 cụm
def clusterDescriptor(descriptors,no_clusters):
    t3 = time.time()
    centroids, _ = kmeans(descriptors, no_clusters, 1)
    # centroids, _
    t4 = time.time()
    print("Done clustering in %d seconds" %(t4-t3))
    return centroids




def extractFeatures(kmeans,descriptor_list,no_clusters,no_images):
    #no_images = len(image_paths)
    im_features = np.zeros((no_images, no_clusters), "float32")
    # im_features[i][j]: số lượng cụm thứ j xuất hiện ở ảnh thứ i
    for i in range(no_images):
        if descriptor_list[i] is not None:
            indexes, distance = vq(descriptor_list[i], kmeans)
            # index cua cluster
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

# list chứa đường dẫn tới các ảnh và class tương ứng
image_paths,image_classes= getFiles(train_path,classes_name)

descriptor_list = getDescriptorList(image_paths)

descriptors = vstackDescriptors(descriptor_list)

# 2. Phân cụm các descriptor
# phân cụm các descriptor
k_centroid = clusterDescriptor(descriptors,no_clusters)

# 3. Xây dựng tập BOW(tần suất xuất hiện của các cụm đã phân cụm ở trên ở từng ảnh)
no_images = len(image_paths)
im_features = extractFeatures(k_centroid,descriptor_list,no_clusters,no_images)

# chuẩn hóa histogram feature về mean = 0 và std = 1
stdSlr,im_features = standardize(im_features)

plotHistogram(im_features, no_clusters)
print("Features histogram plotted.")

# 4. Phân loại
# sử dụng SVM để phân train
# Tìm ra bộ tham số tốt nhất
t5 = time.time()
train_labels = np.array(image_classes)
gsSVM= svcParamSelection(im_features,train_labels ,5)
t6 = time.time()
print("Done classify in %d seconds" %(t6-t5))


print("Best param:"+ gsSVM.best_params_)
print(gsSVM.best_score_)
print( "Training completed")

# lưu lại mô hình
import joblib
joblib.dump((gsSVM.best_estimator_, classes_name, stdSlr, no_clusters, k_centroid), "sift500_coil100.pkl", compress=3)

"""

clf = gsSVM.best_estimator_
test_path = 'coil-100-BOW/test'
test_paths = []
test_classes = []
test_id = 0

print("Start testing")
test_paths,test_classes,test_id = getFiles(test_path,classes_name)
test_descriptor_list = getDescriptorList(test_paths)

test_descriptors = vstackDescriptors(test_descriptor_list)
test_k_centroid = clusterDescriptor(test_descriptors,no_clusters)
# 3. Xây dựng tập BOW(tần suất xuất hiện của các cụm đã phân cụm ở trên ở từng ảnh)
test_im_features = extractFeatures(test_k_centroid,test_descriptor_list,no_clusters)
# standardlize
test_im_features = stdSlr.transform(test_im_features)

pred = clf.predict(test_im_features)
accuracy = accuracy_score(test_classes, pred)
print(accuracy)
    """
