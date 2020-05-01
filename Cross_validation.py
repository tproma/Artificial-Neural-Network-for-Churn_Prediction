
# Cross validation  
from keras.wrappers.scikit_learn import KerasClassifier # wrapper
from sklearn.model_selection import cross_val_score # function
from keras.models import Sequential
from keras.layers import Dense
def model():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 40, units = 20, kernel_initializer = "uniform",  ))
    classifier.add(Dense(activation = 'relu', units = 20, kernel_initializer = "uniform",  ))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = "uniform" ))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = model, batch_size = 10, nb_epoch =100)
accuracies = cross_val_score( estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1 )

mean = accuracies.mean()
variance = accuracies.std()
