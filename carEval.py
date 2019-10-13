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

	# Creating the object for Pre-Processing
	preprocessing_object = preprocessing.LabelEncoder()

	# Reading the data
	data = pd.read_csv(path)

	# Function to convert given categorical column values to numerical values
	def numericalize(self,column_name):

		return preprocessing_object.fit_transform(list(self.data[column_name])) 

	def convertCategoricalToNumerical(self):
		



# Printing the data to check if it worked fine
print(data.head())



# Doing the pre-processing
# this returns a numpy array which is stored in these variables
buying = preprocessing_object.fit_transform(list(data["buying"]))
maint = preprocessing_object.fit_transform(list(data["maint"]))
door = preprocessing_object.fit_transform(list(data["door"]))
persons = preprocessing_object.fit_transform(list(data["persons"]))
lug_boot = preprocessing_object.fit_transform(list(data["lug_boot"]))
safety = preprocessing_object.fit_transform(list(data["safety"]))
car_class = preprocessing_object.fit_transform(list(data["class"]))


# Features
# Makes a list of tuples using zip
# each tuples contains a row entry
features = list(zip(buying, maint, door, persons, lug_boot, safety))

# Labels
labels = list(car_class)

# Splitting the data into testing and training data
# Each one of these being list of tuples which contains entries of one row
train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(features, labels, test_size=0.1)

# Creating model
# takes one parameter : num of neighbors
model = KNeighborsClassifier(n_neighbors=5)

# Training
model.fit(train_features, train_labels)

# Getting the accuracy of our model
accuracy = model.score(test_features, test_labels)

# Printing the accuracy
print(accuracy)

# Predicting
predicted = model.predict(test_features)


# seeing datapoints predictionsand actual vals

names = ["unacc","acc","good","vgood"]

for x in range(len(test_features)):

	print("Predicted : ",names[predicted[x]],"   Data : ",test_features[x], "  Actual : ",names[test_labels[x]])

	# finding 9 neighbors for each data point
	n = model.kneighbors([test_features[x]], 9, True)	

	print("N : ",n)
