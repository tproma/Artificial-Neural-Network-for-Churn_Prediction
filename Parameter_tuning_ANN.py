
# Parameter Tuning 
from keras.wrappers.scikit_learn import KerasClassifier # wrapper
from sklearn.model_selection import GridSearchCV #function
#from sklearn.grid_search import GridSearchCV # function
from keras.models import Sequential
from keras.layers import Dense
def model(optimizer):
   classifier = Sequential()
   classifier.add(Dense(activation = 'relu', input_dim = 40, units = 20, kernel_initializer = "uniform",  ))
   classifier.add(Dense(activation = 'relu', units = 20, kernel_initializer = "uniform",  ))
   classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = "uniform" ))
   classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
   return classifier

classifier = KerasClassifier(build_fn = model)
parameters = {'batch_size': [20,35],
               'epochs': [100,500],
               'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parammeters = grid_search.best_params_
best_accuracy = grid_search.best_score_
