# Artificial Neural Network

#Theano
#Tensorflow
#Keras

#Part 1 : data preprocessing

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])

ct = ColumnTransformer([("encoder", OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

X = X[:, 1:]


#spliting the data into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

#Part 2 : Make ANN
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the Input Layer and First Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation= 'relu', input_dim = 11))

#Adding Second Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation= 'relu'))

#Adding the Output Layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation= 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to the training set
classifier.fit(X_train,y_train, batch_size = 10, epochs = 100)



#Part 3 : Making the Prediction and evoluating the model

#Predicting the Testset result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Accuracy
#84.1

