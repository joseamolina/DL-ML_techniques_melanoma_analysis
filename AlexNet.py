#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:01:12 2019

@author: STUDENT-CIT\r00156440
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import time
import numpy as np
np.random.seed(1000)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

np.random.seed(123)
from PIL import Image
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def draw_confusion_matrix(true, preds):
    conf_matrix = confusion_matrix(true, preds)
    print(conf_matrix)

def prep_submissions(pred_array):
    preds_df = pd.DataFrame(pred_array)
    predicted_labels = preds_df.idxmax(axis=1)
    return predicted_labels

def plot_confusion_matrix(y_true, y_pred, classes):
    title = 'Confusion matrix AlexNet'
    # Compute confusion matrix
    cm = confusion_matrix(y_pred.argmax(axis=1), y_true.argmax(axis=1))
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    cmap=plt.cm.Blues
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def fitting(model, X, val_X, Y, val_Y, EPOCHS, BATCH_SIZE, early_stopping):
    results = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_X, val_Y))

    print("Val Score: ", model.evaluate(val_X, val_Y))
    return results

def cnn_alexnet(n_layers):

    model = Sequential()
    
    model.add(Conv2D(filters=64, input_shape=(75, 100, 3), kernel_size=(3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization())
    
    
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(100*75*3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    model.summary()
    
    return model

base_skin_dir = os.path.join('..', 'input')

# Merge images from both folders into one dictionary
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

print("Obstacle 1")
#Corresponding file containing all information of each image
tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

# Create some new columns (path to image, human-readable name) and review them
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
print(tile_df.sample(5))

# Resizing the image to adjust for parametring to 100-75
tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

# Size of the image to train
input_shape = tile_df['image'].map(lambda x: x.shape).value_counts()
print(input_shape)

print("Obstacle 2")

y = tile_df.cell_type_idx

#x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(tile_df, y, test_size=0.25)

# NEW
model = cnn_alexnet(2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

pat = 5

stopping_cond = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

epochs = 60
batch_size = 32

model_history = []
time_consuming = 0
print("Obstacle 3")

X, val_X, Y, val_Y = train_test_split(tile_df, y, test_size=0.2, random_state=None)

X = np.asarray(X['image'].tolist())
val_X = np.asarray(val_X['image'].tolist())

X_mean = np.mean(X)
X_std = np.std(X)

X_mean_val = np.mean(val_X)
X_std_val = np.std(val_X)

X = (X - X_mean)/X_std
val_X = (val_X - X_mean_val)/X_std_val

Y = to_categorical(Y, num_classes=7)
val_Y = to_categorical(val_Y, num_classes=7)

t1 = time.time()
model_history = fitting(model, X, val_X, Y, val_Y, epochs, batch_size, stopping_cond)
t2 = time.time()

model_json = model.to_json()
with open("model_alex.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model_alex.h5')

time_consuming = t2-t1

print("=======" * 12, end="\n\n\n")

print("Obstacle 4")

plt.title('Accuracy')

plt.plot(model_history.history['acc'], label='Training')
plt.plot(model_history.history['val_acc'], label='Testing')
plt.legend()
plt.show()

print("Obstacle 5")

plt.title('Loss')
plt.plot(model_history.history['loss'], label='Training')
plt.plot(model_history.history['val_loss'], label='Testing')    

plt.legend()
plt.show()
print("Obstacle 6")
print("Time consuming: ", time_consuming)

plot_confusion_matrix(val_Y, model.predict(val_X), tile_df['dx'])

plt.show()

matrix = confusion_matrix(model.predict(val_X).argmax(axis=1), val_Y.argmax(axis=1))
print(matrix)