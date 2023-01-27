import cv2 
import matplotlib.pyplot as plt


#reading image
img1 = cv2.imread('obj1.png')  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.SIFT_create(128)
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

"""
 descriptor
    [[  1.   0.   0. ...   0.   4.  14.]
 [ 37.   0.   0. ...   0.   0.   0.]
 [ 28.   0.   0. ...   0.   0.   0.]
 ...
 [133.  16.   0. ...   4.   5.  30.]
 [  0.   0.   0. ...   0.   0.   0.]
 [ 64.   0.   0. ...   0.   0.   0.]]
 (89,128)  : (keypoint,128)
 len(keypoint) =  89 
 """
print(descriptors_1.shape)
print(len(keypoints_1))
img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
plt.imshow(img_1)