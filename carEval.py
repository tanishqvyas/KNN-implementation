import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing


def debug(x="debugging"):
	print(x)


class KNNclassifier:


	# Declaring file path
	path = os.path.join("Car Data Set","cardata.csv")

	# Variable holding the name of column we wanna predict
	predict_column = "class"

	# seeing datapoints predictionsand actual vals
	names = ["unacc","acc","good","vgood"]

	# accuracy
	accuracy = 0

	# Num of neighbors needed to be checked
	num_of_neighbors = 9

	# Reading the data
	data = pd.read_csv(path)

	# Declaring training and testing features and labels list
	train_features = []
	test_features = [] 
	train_labels = [] 
	test_labels = []

	def __init__(self):


		self.features_and_labels()

		self.split_into_test_train_data()

		self.make_train_predict()

	# Function to convert given categorical column values to numerical values
	def convertCategoricalToNumerical(self,column_name):

		# Creating the object for Pre-Processing
		preprocessing_object = preprocessing.LabelEncoder()
	
		return preprocessing_object.fit_transform(list(self.data[column_name])) 

	def features_and_labels(self):
		
		# Doing the pre-processing
		# this returns a numpy array which is stored in these variables
		buying = self.convertCategoricalToNumerical("buying")
		maint = self.convertCategoricalToNumerical("maint")
		door = self.convertCategoricalToNumerical("door")
		persons = self.convertCategoricalToNumerical("persons")
		lug_boot = self.convertCategoricalToNumerical("lug_boot")
		safety = self.convertCategoricalToNumerical("safety")
		car_class = self.convertCategoricalToNumerical("class")

		# Features
		# Makes a list of tuples using zip
		# each tuples contains a row entry

		# Labels

		return list(zip(buying, maint, door, persons, lug_boot, safety)) , list(car_class)


	def split_into_test_train_data(self):

		# Getting the features and labels for our dataset
		features, labels = self.features_and_labels()


		# Splitting the data into testing and training data
		# Each one of these being list of tuples which contains entries of one row
		self.train_features, self.test_features, self.train_labels, self.test_labels = sklearn.model_selection.train_test_split(features, labels, test_size=0.1)


	def make_train_predict(self):

		# Creating model
		# takes one parameter : num of neighbors
		model = KNeighborsClassifier(n_neighbors=self.num_of_neighbors)

		# Training
		model.fit(self.train_features, self.train_labels)

		# Getting the accuracy of our model
		self.accuracy = model.score(self.test_features, self.test_labels)

		# Printing the accuracy
		print(self.accuracy)

		# Predicting
		predicted = model.predict(self.test_features)

	
		
		for x in range(len(self.test_features)):
		
			print("Predicted : ",self.names[predicted[x]],"   Data : ",self.test_features[x], "  Actual : ",self.names[self.test_labels[x]])
		
			# finding 9 neighbors for each data point
			#n = model.kneighbors([test_features[x]], 9, True)	
			#print("N : ",n)



if __name__ == "__main__":
	classifier_object = KNNclassifier()








