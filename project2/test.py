import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from scipy.cluster.vq import vq
import joblib
import time
from matplotlib import pyplot as plt

clf, classes_name, stdSlr, no_clusters, k_centroid = joblib.load("sift500_coil100.pkl")
def plotConfusionMatrix(y_true, y_pred, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    
    title = 'Confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    
    print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

test_path = 'image/test'
test_paths = []
test_classes = []
test_id = 0


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

print("Start testing")
test_paths,test_classes= getFiles(test_path,classes_name)
x = np.array(test_classes)
classes = np.unique(x)
print(classes)
test_descriptor_list = getDescriptorList(test_paths)



# 3. Xây dựng tập BOW(tần suất xuất hiện của các cụm đã phân cụm ở trên ở từng ảnh)
no_images = len(test_paths)
test_im_features = extractFeatures(k_centroid,test_descriptor_list,no_clusters,no_images)
# standardlize
test_im_features = stdSlr.transform(test_im_features)

pred = clf.predict(test_im_features)
accuracy = accuracy_score(test_classes, pred)

plotConfusionMatrix(test_classes, pred,classes)
plt.show()
print("Accuracy:"+ str(accuracy*100)+ "%")
