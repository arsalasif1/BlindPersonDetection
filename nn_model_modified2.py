#nural network try
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Embedding
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten



# load the dataset
df = pd.read_csv('coords.csv')
print(df.size)

df[df['class']=='Blind']

X = df.drop('class', axis=1) # features
#print(X)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X=imp.transform(X)

y = df['class'] # target value
y=(y=='Blind')*1.0
#print(y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=1234)
#model = Sequential()
#model.add(Embedding(50, 32, input_length=132))
#model.add(Flatten())
#model.add(Dense(250, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()
## Fit the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=10, verbose=1)
## Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))


#dataset=dataset[~np.isnan(dataset).any(axis=1)]
#dataset = normalize(dataset, axis=0, norm='max')
# split into input (X) and output (y) variables
#X = np.hstack([dataset[:,0:3],dataset[:,5:7]])
#X=dataset[:,0:9]
#y = dataset[:,9]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
# define the keras model
model = Sequential()
model.add(Dense(5, input_shape=(132,), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=10, verbose=1)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
model.save('model')
