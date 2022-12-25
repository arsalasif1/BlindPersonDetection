#nural network try
#from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import normalize
# load the dataset
dataset = loadtxt('foo_onesample.csv', delimiter=',')
dataset=dataset[~np.isnan(dataset).any(axis=1)]
dataset = normalize(dataset, axis=0, norm='max')
# split into input (X) and output (y) variables
#X = np.hstack([dataset[:,0:3],dataset[:,5:7]])
X=dataset[:,0:9]
y = dataset[:,9]
# define the keras model
model = Sequential()
model.add(Dense(50, input_shape=(9,), activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=10, verbose=1)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
model.save('/home/arsal/Desktop/MLproject/RNN/model')
