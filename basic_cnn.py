import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
#import sns as sns
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from skimage.feature import hog
from sklearn.model_selection import KFold
from sklearn import svm
from skimage import color
from matplotlib import pyplot as plt

np.random.seed(123)
from PIL import Image
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

#Dict of options
GREY_SCALE = True

def draw_confusion_matrix(true, preds):
    conf_matrix = confusion_matrix(true, preds)
    print(conf_matrix)
    #sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 12}, fmt='g', cbar=False, cmap='viridis')
    #plt.show()

def prep_submissions(pred_array):
    preds_df = pd.DataFrame(pred_array)
    predicted_labels = preds_df.idxmax(axis=1)
    return predicted_labels

def fitting_cnn_simple(model, X, val_X, Y, val_Y, EPOCHS, BATCH_SIZE, early_stopping):
    
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    print(X.shape, Y.shape)
    print(val_X.shape, val_Y.shape)
    
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    results = model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], validation_data=(val_X, val_Y))

    print("Validation Score: ", model.evaluate(val_X, val_Y))
    print("Training Score: ", model.evaluate(X, Y))
    return results

def cnn_architecture1(n_layers):

    size = (75, 100, 3)
    NUM_FILTERS = 32
    num_classes = 7
    KERNEL = (3, 3)
    
    model = Sequential()
    
    for i in range(1, n_layers - 2):
        if i == 1:
            model.add(Conv2D(NUM_FILTERS * i, kernel_size=(3, 3), input_shape=size, activation='relu'))
        else:
            model.add(Conv2D(NUM_FILTERS * i, KERNEL, activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cnn_architecture2(size):

    nun_classes = 7

    model = Sequential()

    model.add(Conv2D(32, kernel_size= (3, 3), activation='relu', padding='Same', input_shape=size))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nun_classes, activation='softmax'))
    
    return model

print("Hola mundo!\n")

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

tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

# Size of the image to train. Important in input shape
input_shape = tile_df['image'].map(lambda x: x.shape).value_counts()

print(input_shape)
print("Obstacle 2")

y = tile_df.cell_type_idx

for n_lay in range(4, 12):
    
    pat = 12
    stopping_cond = EarlyStopping(monitor='loss', patience=pat, verbose=1)
    
    n_folds = 4
    epochs = 30
    batch_size = 32
    
    print("Obstacle 3")
    
    model = cnn_architecture1(n_lay)
    
    model.build()
    print(model.summary())
    
    X, val_X, Y, val_Y = train_test_split(tile_df, y, test_size = 0.3)
    
    X = np.asarray(tile_df['image'].tolist())
    
    val_X = np.asarray(tile_df['image'].tolist())

    X_mean = np.mean(X)
    X_std = np.std(X)

    X_mean_val = np.mean(val_X)
    X_std_val = np.std(val_X)

    X = (X - X_mean)/X_std
    val_X = (val_X - X_mean_val)/X_std_val

    Y = to_categorical(Y, num_classes=7)
    val_Y = to_categorical(val_Y, num_classes=7)

    model_history = fitting_cnn_simple(model, X, val_X, Y, val_Y, epochs, batch_size, stopping_cond)

    print("=======" * 12, end="\n\n\n")
    
    plt.title('Accuracy')
    
    plt.plot(model_history.history['acc'], label='Training')
    plt.plot(model_history.history['val_acc'], label='Testing')

    plt.legend()
    plt.show()
    
    plt.title('Loss')
    
    plt.plot(model_history.history['loss'], label='Training')
    plt.plot(model_history.history['val_loss'], label='Testing')

    plt.legend()
    plt.show()

"""
plt.legend()
plt.show()
print("Obstacle 6")
model.evaluate(x_test_o, y_test_o)

tests_pred = model.predict(x_test_o)
tests_preds_labels = prep_submissions(tests_pred)

print(classification_report(tile_df['cell_type_idx'] , tests_preds_labels))

draw_confusion_matrix(tile_df['cell_type_idx'], tests_preds_labels)
"""