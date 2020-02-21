import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from skimage import color
import pandas as pd
from skimage.feature import hog
from sklearn import svm
import os
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split



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



tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))
tile_df['image'] = tile_df['image'].map(lambda x: color.rgb2gray(x))

plt.imshow(tile_df['image'][4])

#.resize((100,75))

# Resizing the image to adjust for parametring to 100-75
#tile_df['image'] = tile_df['image'].map(lambda x: x.resize((100, 75)))

# Resizing the image to adjust for parametring to 100-75
#tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))

plt.imshow(tile_df['image'][4])

# Size of the image to train. Important in input shape
input_shape = tile_df['image'].map(lambda x: x.shape).value_counts()


print(input_shape)
print("Obstacle 2")

ppc = 16
hog_images = []

hog_features = []

for image in tile_df['image']:
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualise=True)
    hog_images.append(hog_image)
    hog_features.append(fd)

plt.imshow(hog_images[51])
hog_features = np.array(hog_features)

y = tile_df.cell_type_idx

#set_data = np.stack((hog_features, y))


x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(hog_features, y, test_size=0.25)

print(x_train_o, x_test_o, y_train_o, y_test_o)


clf = svm.SVC()

clf.fit(x_train_o, y_train_o)

y_pred = clf.predict(x_test_o)

print(accuracy_score(y_test_o, y_pred))

print(classification_report(y_test_o, y_pred))
