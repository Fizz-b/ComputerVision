from project2.init import initFile,initFolder
import cv2
initFolder()
initFile()
# ```console
# python train.py --train_path image/train/ --test_path image/test/
#```
import numpy as np 
from glob import glob 
import argparse
from helper import *
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix

def plotConfusionMatrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    title = 'Confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

class BOW:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bow_helper = BOWHelpers(no_clusters)
        self.no_clusters = no_clusters
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        """
        This method contains the entire module 
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        #1.extract SIFT Features from each image
        label_count = 0 
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print ("Computing Features for ", word)
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()
                self.train_labels = np.append(self.train_labels, label_count)
                keypoint, descriptor = self.im_helper.sift_features(im)
                self.descriptor_list.append(descriptor)

            label_count += 1

        
        
        # 2.perform clustering
        bow_descriptor_stack = self.bow_helper.formatND(self.descriptor_list)
        self.bow_helper.cluster()
        self.bow_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)
           
       
 

        self.bow_helper.standardize()
        # show vocabulary trained
        self.bow_helper.plotHist()
        # train using grid search to find param
        self.bow_helper.train(self.train_labels)


    def recognize(self,test_img, test_image_path=None):

        """ 
        This method recognizes a single image 
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.sift_features(test_img)
        # print kp
        print (des.shape)

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bow_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1

        print (vocab)
        # Scale the features
        vocab = self.bow_helper.scale.transform(vocab)

        # predict the class of the image
        lb = self.bow_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

  

    def testModel(self):
        """ 
        This method is to test the trained classifier

        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)
        
        predictions = []
        des_list = []
        test_classes={}
        
        result_classes=[]
          #1.extract SIFT Features from each image
        label_count = 0 
        for word, imlist in self.testImages.items():
            test_classes[str(label_count)] = word
            for im in imlist:
                result_classes.append(word)
                keypoint, descriptor = self.im_helper.sift_features(im)
                des_list.append(descriptor)

            label_count += 1

        
        
        # 2.perform clustering
        bow_helper = BOWHelpers(self.no_clusters)
        bow_descriptor_stack = bow_helper.formatND(des_list)
        bow_helper.cluster()
        bow_helper.developVocabulary(n_images = self.testImageCount, descriptor_list=des_list)
        bow_helper.plotHist() 
        bow_helper.standardize()
        bow_helper.clf = self.bow_helper.clf
        predictions = bow_helper.predict(bow_helper.mega_histogram)
        
        print(result_classes)
        print(predictions)
        result=[]
        for prediction in predictions:
            result.append(self.name_dict[str(int(prediction))])
        print(result)  # output
        #result_classes -actual predict
        
        
        accuracy = accuracy_score(result_classes, result)
        print(accuracy)
        
        plotConfusionMatrix(result_classes, result,classes=self.name_dict.values())
        plt.show()
        print("Confusion matrixes plotted.")
         
        """
            
        for word, imlist in self.testImages.items():
            print ("processing " ,word)
            for im in imlist:
                # print imlist[0].shape, imlist[1].shape
                print( im.shape)
                cl = self.recognize(im)
                print (cl)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })
                    
                         print (predictions)
        for each in predictions:
            # cv2.imshow(each['object_name'], each['image'])
            # cv2.waitKey()
            # cv2.destroyWindow(each['object_name'])
            # 
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()
            """
       




if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words "
        )
    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)

    args =  vars(parser.parse_args())
    print(args)

    
    bow = BOW(no_clusters=100)

    # set training paths
    bow.train_path = args['train_path'] 
    # set testing paths
    bow.test_path = args['test_path'] 
    # train the model
    bow.trainModel()
    # test model
    bow.testModel()


