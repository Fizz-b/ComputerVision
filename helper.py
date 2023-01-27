import cv2
import numpy as np 
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class ImageHelpers:
	def __init__(self):
		self.sift_object = cv2.SIFT_create()

	def gray(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return gray

	def sift_features(self, image):
		keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
		return [keypoints, descriptors]
		


class BOWHelpers:
	def __init__(self, n_clusters = 20):
		self.n_clusters = n_clusters
		self.kmeans_obj = KMeans(n_clusters = n_clusters)
		self.kmeans_ret = None
		self.descriptor_vstack = None
		self.mega_histogram = None
		self.clf  =  None
  
	def svcParamSelection(self,X, y, nfolds):
		"""	
		param selection 

		"""
		Cs = [0.1, 1, 10, 100, 1000]
		gammas = [1, 0.1, 0.01, 0.001, 0.0001]
		param_grid = {'C': Cs, 'gamma' : gammas}
		grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
		grid_search.fit(X, y)
		return grid_search.best_params_
	def cluster(self):
		"""	
		cluster using KMeans algorithm, 

		"""
		print("Start cluster")
		#self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)
		self.kmeans_ret = self.kmeans_obj.fit(self.descriptor_vstack)

	def developVocabulary(self,n_images, descriptor_list, kmeans_ret = None):
		
		"""
		Each cluster denotes a particular visual word 
		Every image can be represeted as a combination of multiple 
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word 

		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images

		"""
		print("Build vocab")
		self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
		"""
		old_count = 0
		for i in range(n_images):
				l = len(descriptor_list[i])
				for j in range(l):
					if kmeans_ret is None:
						idx = self.kmeans_ret[old_count+j]
					else:
						idx = kmeans_ret[old_count+j]
					self.mega_histogram[i][idx] += 1
				old_count += l
		"""
		for i in range(n_images):
			for j in range(len(descriptor_list[i])):
				feature = descriptor_list[i][j]
				feature = feature.reshape(1, 128)
				if kmeans_ret is None:
						idx = self.kmeans_ret.predict(feature)
				else:
						idx = kmeans_ret.predict(feature)
				self.mega_histogram[i][idx] += 1
		print( "Vocabulary Histogram Generated")

	def standardize(self, std=None):
		"""
		
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.

		"""
		print("Standardlize ")
		if std is None:
			self.scale = StandardScaler().fit(self.mega_histogram)
			self.mega_histogram = self.scale.transform(self.mega_histogram)
		else:
			print( "STD not none. External STD supplied")
			self.mega_histogram = std.transform(self.mega_histogram)
    
    # stack descriptor de phan cum
    # in : descriptor list
	def formatND(self, descriptor_list):
		"""	
		restructures list into vstack array of shape
		M samples x N features for sklearn

		"""
		vStack = np.array(descriptor_list[0])
		#vStack = np.array(l[0][1])
		# except first 
		for remaining in descriptor_list[1:]:
			if remaining is not None:
				vStack = np.vstack((vStack, remaining))
		self.descriptor_vstack = vStack.copy()
		return vStack

	def train(self, train_labels):
		"""
		uses sklearn.svm.SVC classifier (SVM) 
		"""
		print( "Training SVM")
		print( self.clf)
		print( "Train labels", train_labels)
        #self.clf.fit(self.mega_histogram, train_labels)
        
        # k fold find optimal hyperparameter
		params = self.svcParamSelection(self.mega_histogram, train_labels, 5)
		C_param, gamma_param = params.get("C"), params.get("gamma")
		
		self.clf = SVC(C =  C_param, gamma = gamma_param)
		self.clf.fit(self.mega_histogram, train_labels)
		print(C_param, gamma_param)
		print( "Training completed")

	def predict(self, test_features):
		predictions = self.clf.predict(test_features)
		return predictions
	
	
	def plotHist(self, vocabulary = None):
		print( "Plotting histogram")
		if vocabulary is None:
			vocabulary = self.mega_histogram

		x_scalar = np.arange(self.n_clusters)
		y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.n_clusters)])

		print( y_scalar)

		plt.bar(x_scalar, y_scalar)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x_scalar + 0.4, x_scalar)
		plt.show()

class FileHelpers:

	def __init__(self):
		pass

	def getFiles(self, path):
		"""
		- returns  a dictionary of all files 
		having key => value as  objectname => image path

		- returns total number of files.

		"""
		imlist = {}
		count = 0
		for each in glob(path + "*"):
			word = each.split("/")[-1]
			obj = word.split("\\")[-1]
			print( " #### Reading image category ", obj, " ##### ")
			imlist[obj] = []
			for imagefile in glob(path+obj+"/*"):
				print( "Reading file ", imagefile)
				im = cv2.imread(imagefile, 0)
				imlist[obj].append(im)
				count +=1 
        # list for each obj : imlist[obj1] =  ["img1","img2"]
		return [imlist, count]

