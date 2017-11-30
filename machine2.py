import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from numpy import loadtxt

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
      Y[n] = [1,0,0];
   if i=='Iris-versicolor':
      Y[n] = [0,1,0];
   if i=='Iris-virginica':
      Y[n] = [0,0,1];

Y = numpy.array(Y)
Y = Y.astype(numpy.integer)
dummy_y = np_utils.to_categorical(Y)


print Y, Yshape



##Machine learning model
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

## the last model Classifier Compile model
##model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)



kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
