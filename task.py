#task

# importing libraries  
import numpy as np  
import matplotlib.pyplot as mtp  
import pandas as pd  

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
#importing datasets  
data_set= pd.read_csv('municipality_bus_utilization.csv')  

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, 1:].values  
y= data_set.iloc[:, 1].values  


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.30, random_state=42)  
  
# Fitting K-NN classifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(x_train, y_train)  

#Fitting SVM classifier
# classifier = SVC(kernel='linear', random_state=0)  
# classifier.fit(x_train, y_train) 

#Predicting the test set result  
y_pred= classifier.predict(x_test)
print(y_pred)
# # accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
