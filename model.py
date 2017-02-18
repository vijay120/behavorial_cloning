import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import csv
import math

from keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.models import load_model

mainDir = '/home/carnd/data/'
secondaryDir = '/home/carnd/more_data_2/'

df = pd.read_csv(mainDir + 'driving_log.csv', sep=',', header=0)

## data I obtained by driving around the lap
df2 = pd.read_csv(secondaryDir + 'driving_log.csv', sep=',', header=0)

## combine the two datasets
df = df.append(df2, ignore_index=True)

inbet = df[(df["steering"] < 0.05) & (df["steering"] > -0.05)]
out = df[(df["steering"] > 0.05) | (df["steering"] < -0.05)]

## remove data with small steering angels
df = pd.concat([out, inbet.sample(frac=0.5, replace=False)])

train_samples, validation_samples = train_test_split(df, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            X = batch_samples["center"]
            Left = batch_samples["left"]
            Right = batch_samples["right"]
            
            X_Regular = X.map(lambda img : cv2.imread(mainDir + img.strip().replace("/Users/vijaytramakrishnan/more_data_2/", ""), 1))
            Y_Regular = batch_samples["steering"]

            X_Left = Left.map(lambda img : cv2.imread(mainDir + img.strip().replace("/Users/vijaytramakrishnan/more_data_2/", ""), 1))
            Y_Left = batch_samples["steering"] + 0.2

            X_Right = Right.map(lambda img : cv2.imread(mainDir + img.strip().replace("/Users/vijaytramakrishnan/more_data_2/", ""), 1))
            Y_Right = batch_samples["steering"] - 0.2

            X_Reg_flipped = X.map(lambda img : cv2.flip(cv2.imread(mainDir + img.strip().replace("/Users/vijaytramakrishnan/more_data_2/", ""), 1), 1))
            Y_Reg_flipped = Y_Regular.map(lambda measurement: -1.0 * measurement)

            X_Left_flipped = Left.map(lambda img : cv2.flip(cv2.imread(mainDir + img.strip().replace("/Users/vijaytramakrishnan/more_data_2/", ""), 1), 1))
            Y_Left_flipped = Y_Left.map(lambda measurement: -1.0 * measurement)

            X_Right_flipped = Right.map(lambda img : cv2.flip(cv2.imread(mainDir + img.strip().replace("/Users/vijaytramakrishnan/more_data_2/", ""), 1), 1))
            Y_Right_flipped = Y_Right.map(lambda measurement: -1.0 * measurement)

            X_concat = pd.concat([X_Regular, X_Left, X_Right, X_Reg_flipped, X_Left_flipped, X_Right_flipped])
            Y_concat = pd.concat([Y_Regular, Y_Left, Y_Right, Y_Reg_flipped, Y_Left_flipped, Y_Right_flipped])
            X_concat = np.asarray(X_concat.tolist())
            Y_concat = np.asarray(Y_concat.tolist())
                        
            yield shuffle(X_concat, Y_concat)    
    
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

dropout = 0.2
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))


model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile('adam', 'mse')
model.fit_generator(train_generator, 
                    samples_per_epoch=len(train_samples), 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=7)

model.save('model.h5') 