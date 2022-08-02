import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.express as px

sns.set_theme(color_codes=True)
sns.set_style('whitegrid')
init_notebook_mode(connected=True)
cf.go_offline()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Convolution2D(64, 3,3, input_shape=(200,200,3), activation=('relu')))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 3,3, activation=('relu')))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics=(['accuracy']))


#IMAGE PREPROCESSING

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,                                   
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',                                                 
                                                 target_size=(200, 200),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',                                                        
                                            target_size=(200, 200),
                                            batch_size=32,
                                            class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000/32,
        epochs=50,
        validation_data=test_set,
        validation_steps=2000/32)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_function(optimizer):
    model = Sequential()

    model.add(Convolution2D(64, 3,3, input_shape=(128,128,3), activation=('relu')))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64, 3,3, input_shape=(128,128,3), activation=('relu')))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=(['accuracy']))
    return model
    
fn_model = KerasClassifier(build_fn=(build_function))
parameters = {'epochs' :[25,30,40],
              'optimizer' : ['adam','rmsprop']}

grid_model = GridSearchCV(estimator = fn_model, param_grid = parameters, cv=10, scoring=('accuracy'))

grid_model.fit(training_set,test_set)



from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

model.predict_classes(test_image)
training_set.class_indices





