import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]


# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_Y)

for n,i in enumerate(Y):
   if i=='Iris-setosa':
      Y[n] = 0;
   if i=='Iris-versicolor':
      Y[n] = 1;
   if i=='Iris-virginica':
      Y[n] = 2;

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='sigmoid'))
	model.add(Dense(3, init='normal'))
        model.add(Activation('softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

## the last model Classifier Compile model
##model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)


# evaluate the model
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

## text formatting% (return, return)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




