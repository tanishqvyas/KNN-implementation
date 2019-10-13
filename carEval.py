import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing


# Declaring file path
path = os.path.join("Car Data Set","cardata.csv")

# Reading the data
data = pd.read_csv(path)

# Printing the data to check if it worked fine
print(list(data.head()))

# Variable holding the name of column we wanna predict
predict = "class"

# Creating the object for Pre-Processing
preprocessing_object = preprocessing.LabelEncoder()

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
# Makes a list of lists using zip
features = list(zip(buying, maint, door, persons, lug_boot, safety))

# Labels
labels = list(car_class)

# Splitting the data into testing and training data
# Each one of these being list of tuples which contains entries of one row
train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(features, labels, test_size=0.1)
#print(train_features)