#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:31:56 2017

@author: Nikos Chlis
"""
import numpy as np
import matplotlib.pyplot as plt
import time
#from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from keras.datasets import cifar10
(X_tr, y_tr), (X_ts, y_ts) = cifar10.load_data()


#map numeric labels to class names
#source: https://www.cs.toronto.edu/~kriz/cifar.html
classes=np.unique(y_tr)
num_classes=len(classes)#number of classes
classes_char=['airplane','automobile','bird','cat','deer',
              'dog','frog','horse','ship','truck']

#plot an example image from each class
fig=plt.figure(figsize=(3,6))
plt.suptitle('CIFAR 10 classes')
for i in np.arange(num_classes):
#    plt.figure(figsize=(30,30))
    pic=(np.where((y_tr==i))[0])[0]
    ax=fig.add_subplot(num_classes/2,2,i+1)
    ax.imshow(X_tr[pic,:,:])
    ax.set_title(str(i)+': '+classes_char[i])
fig.tight_layout()
fig.subplots_adjust(top=0.9)#to show suptitle properly
fig.show()
plt.savefig('CIFAR10_all_categories.png',dpi=300)

#normalize input images to [0,1]
X_tr=X_tr/2**8
X_ts=X_ts/2**8

#convert y to categorical
y_tr = np_utils.to_categorical(y_tr, num_classes)
y_ts = np_utils.to_categorical(y_ts, num_classes)

#%% set up a VGG-like model
# architecture inspired by http://torch.ch/blog/2015/07/30/cifar.html
nfilters = [64,128,256]#number of filters of convs in each Conv block
ndense = 512#number of units in each fully connected layer
add_BatchNorm = True
dropout_rate_conv = 0.2
dropout_rate_dense = 0.5

model_id='CNN_bn_'+str(add_BatchNorm)+'_dropConv_'+str(dropout_rate_conv)+'_dropDense_'+str(dropout_rate_dense)
print('Build model...',model_id)

model = Sequential()

#Conv block #1
model.add(Conv2D(nfilters[0], (3, 3), padding='same',
                 input_shape=X_tr.shape[1:]))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_conv))

model.add(Conv2D(nfilters[0], (3, 3)))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_conv))

model.add(MaxPooling2D(pool_size=(2, 2)))

#Conv block #2
model.add(Conv2D(nfilters[1], (3, 3), padding='same'))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_conv))

model.add(Conv2D(nfilters[1], (3, 3)))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_conv))

model.add(MaxPooling2D(pool_size=(2, 2)))

#Conv block #3
model.add(Conv2D(nfilters[2], (3, 3), padding='same'))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_conv))

model.add(Conv2D(nfilters[2], (3, 3)))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_conv))

model.add(MaxPooling2D(pool_size=(2, 2)))

#at this point each image has shape (None, 2, 2, nfilters[2])
model.add(Flatten())
#at this point each image has shape (None, 2*2*nfilters[2])

model.add(Dense(ndense))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_dense))

model.add(Dense(ndense))
if(add_BatchNorm==True):
    model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate_dense))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#%% train the model

#log training progress
csvlog = CSVLogger(model_id+'_train_log.csv',append=True)

#save best model according to validation accuracy
checkpoint = ModelCheckpoint(model_id+'.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

max_epochs=250

tic=time.time()
hist=model.fit(X_tr, y_tr,
                 validation_data=(X_ts, y_ts),
                 epochs=max_epochs, batch_size=64, verbose=2,
                 initial_epoch=0,callbacks=[checkpoint, csvlog])
toc=time.time()

#save final model, in case training for mor than max_epochs is necessary
model.save(model_id+'_'+str(max_epochs)+'_epochs.hdf5')
file = open(model_id+'_time.txt','w')
file.write('training time:'+format(toc-tic, '.2f')+'seconds')
file.close()














