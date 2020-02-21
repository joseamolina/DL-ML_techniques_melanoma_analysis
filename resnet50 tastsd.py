import os
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
import time

from keras.utils.np_utils import to_categorical

import pandas as pd
import numpy as np
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split

#1. Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, classes):
    title = 'Confusion matrix AlexNet'
    # Compute confusion matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
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

resnet_weights_path = '../resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
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

X_train, X_test, Y_train, Y_test = train_test_split(tile_df, y, test_size=0.25)

X_train = np.asarray(X_train['image'].tolist())
X_test = np.asarray(X_test['image'].tolist())

X_mean = np.mean(X_train)
X_std = np.std(X_train)

X_mean_val = np.mean(X_test)
X_std_val = np.std(X_test)

X_train = (X_train - X_mean)/X_std
X_test = (X_test - X_mean_val)/X_std_val

Y_train = to_categorical(Y_train, num_classes=7)
Y_test = to_categorical(Y_test, num_classes=7)

model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=None))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))

model.layers[0].trainable = True


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

stopping_cond = EarlyStopping(monitor='val_loss', verbose=1)

n_epochs = 2
batch_size = 256

model_history = []
testing_history = []
time_consuming = 0

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
model_history = model.fit(X, Y, epochs=n_epochs, batch_size=batch_size, callbacks=[stopping_cond], validation_data=(val_X, val_Y))
t2 = time.time()

time_consuming = t2-t1

val_Y = to_categorical(val_Y, num_classes=7)

(loss, acc) = model.evaluate(X_test, Y_test)

print('Validation prediction: ', (loss, acc))

tests_pred = model.predict(val_X)

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
    
plt.legend()
plt.show()

plot_confusion_matrix(val_Y, model.predict(val_X), tile_df['dx'])

plt.show()

matrix = confusion_matrix(model.predict(val_X).argmax(axis=1), val_Y.argmax(axis=1))
print(matrix)
