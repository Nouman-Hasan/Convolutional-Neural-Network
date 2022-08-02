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

model.add(Convolution2D(64,3,3, input_shape=(128,128,3), activation=('relu')))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3, activation=('relu')))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation=('relu')))
model.add(Dense(3, activation=('softmax')))

model.compile(optimizer='adam', loss=('categorical_crossentropy'), metrics=(['accuracy']))

#IMAGE_PREPROCESSING
from keras.preprocessing.image import ImageDataGenerator
training_datagen = ImageDataGenerator(
                                        rescale = 1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = training_datagen.flow_from_directory('Rock-Paper-Scissors/train',                                                 
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Rock-Paper-Scissors/test',                                                        
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

model.fit(training_set,
          steps_per_epoch=(2520/32),
          epochs=30,
          validation_data=(test_set),
          validation_steps=(372/32))

model.summary()

losses = pd.DataFrame(model.history.history)
losses.plot()





