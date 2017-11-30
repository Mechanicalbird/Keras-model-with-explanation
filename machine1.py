##import the classes from keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.layers.core import TimeDistributedDense, Activation, Dropout  
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import RMSprop

##import fundamental package 
import numpy
import pandas

from numpy import loadtxt

maxlen = 150

batch_size = 5
nb_word = 4
nb_tag = 150

## generate a random numbers
seed = 7
numpy.random.seed(seed)

##get the data

dataframe   = pandas.read_csv("iris.data.csv", header=None)
dataset     = dataframe.values
X = dataset[:,0:4].astype(float)

## process the Y-data
Y = dataset[:,4]
Yshape = numpy.shape(Y)

for n,i in enumerate(Y):
   if i=='Iris-setosa':
      Y[n] = "1";
   if i=='Iris-versicolor':
      Y[n] = "2";
   if i=='Iris-virginica':
      Y[n] = "3";

Y = Y.astype(numpy.float)

print Y, Yshape

##Machine learning model
#model = Sequential()
#model.add(Dense(12, input_dim=4, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='sigmoid'))
#--does not work--#model.add(Activation('softmax'))
model = Sequential()
model.add(Embedding(nb_word, 128))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributedDense(nb_tag))
model.add(Activation('softmax'))



## Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

## Fit the model
#model.fit(X, Y, nb_epoch=40, batch_size=5)
model.fit(X, Y, batch_size=batch_size, nb_epoch=40, show_accuracy=True)

## evaluate the model
scores = model.evaluate(X, Y)


## text formatting% (return, return)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))













