# simple-deeplearning-project

#### Part A and B
##### importing important libraries 
import pandas as pd
import numpy as np

##### import Data
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')

concrete_data.head()

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength

target = concrete_data['Strength'] # Strength column


predictors_norm = (predictors - predictors.mean()) / predictors.std()

predictors_norm.head()

##### deep learning via keras 
import keras

from keras.models import Sequential

from keras.layers import Dense 

##### define regression model
def regression_model():

##### create model
    
   model = Sequential()
    
   model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    
   model.add(Dense(1))
    
##### compile model
 
   model.compile(optimizer='adam', loss='mean_squared_error')
   
   return model
    
    
##### build the model
 
model = regression_model()

##### fit the model

model.fit(predictors_norm, target, validation_split=0.3, epochs=50, verbose=2)

##### mean squared error list 

list_mse = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, shuffle= True, random_state=50, test_size= 0.3)
    
   train_model = linear_nn()
    
   train_model.fit(X_train, y_train, epochs= 50, verbose= 2)
    
   test_predict = train_model.predict(X_test)
    
   mse = mean_squared_error(y_test, test_predict)
    
   list_mse.append(mse)
    
##### print the list

std_mse = np.std(list_mse)

mean_mse =np.mean(list_mse)

print('Mean of MSE: {}'.format(mean_mse))

print('Standard Deviation of MSE: {}'.format(std_mse))


### Part C

##### define regression model

def regression_model():

##### create model
   
   model = Sequential()
   
   model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
   
   model.add(Dense(1))
    
##### compile model

   model.compile(optimizer='adam', loss='mean_squared_error')
   
   return model

##### build the model

model = regression_model()

##### 100 epochs instead of 50 

model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

std_mse = np.std(list_mse)

mean_mse =np.mean(list_mse)

print('Mean of MSE: {}'.format(mean_mse))

print('Standard Deviation of MSE: {}'.format(std_mse))


### Part D

##### define regression model

def regression_model():

##### create model

   model = Sequential()
   
   model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    
   model.add(Dense(10, activation='relu'))
    
   model.add(Dense(10, activation='relu'))
    
   model.add(Dense(1))
    
##### compile model

model.compile(optimizer='adam', loss='mean_squared_error')

return model

##### build the model

model = regression_model()

model.fit(predictors_norm, target, validation_split=0.3, epochs=50, verbose=2)

std_mse = np.std(list_mse)

mean_mse =np.mean(list_mse)

print('Mean of MSE: {}'.format(mean_mse))

print('Standard Deviation of MSE: {}'.format(std_mse))
