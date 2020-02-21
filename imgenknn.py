#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:43:46 2019

@author: Jose Angel Molina
"""
import os
import pandas as pd
import numpy as np

from PIL import Image
from skimage import color

class GenAlgs():
    
    def __init__(self, pool, prof_avg):
        
        self.pool = pool
        self.prof_avg = prof_avg.flatten()
        self.nearest()
    
    def compute_fitness(self, ind):
        
        fitness = 0
        for i in range(100 * 75):
            fitness += self.prof_avg[i] * ind[i]
            
        return fitness
    
    def get_done(self):
        return self.pool
    
    def nearest(self):
        
        while len(self.pool) > 3:
            
            print(len(self.pool))
            self.pooling()
                
    def pooling(self):
        
        print(len(self.pool))
        
        for i in range(0, len(self.pool) // 2, 2):
            
            # Select 2 individuals
            ind1 = list(self.pool[i])
            ind2 = list(self.pool[i+1])

            # Apply crossover
            
            flipA = [0]
            
            firstA = ind1[0]
            goner = ind2[0]
            
            while goner != firstA:
                
                index_new = ind1.index(goner)
                goner = ind2[index_new]#
                
                flipA.append(index_new)
            
            # Creation of 2 individuals
            
            son1 = ind1[:]
            son2 = ind2[:]
            
            for i in flipA:
                son1[i] = ind1[i]
                son2[i] = ind2[i]
                
            self.pool.append(np.array(son1))
            self.pool.append(np.array(son2))
            
        # No mutation needed
        
        permanent = list(map(lambda x: self.compute_fitness(x), self.pool))
        
        for rep in range(3):
            if len(self.pool) != 0:
                min_ind = permanent.index(min(permanent))
                del self.pool[min_ind]

base_skin_dir = os.path.join('..', 'input')

tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

path = '/home/local/STUDENT-CIT/r00156440/PROJECT/input/HAM10000_images_part_1_new'

fils = []
for _, _, z in os.walk(path):
    fils = z
    
fils = pd.Series(fils).apply(lambda x: x[:-4])

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

dir_to_use1 = '/home/local/STUDENT-CIT/r00156440/PROJECT/input/HAM10000_images_part_1_new/'
# It contains the images to get into account for the experiments
new_file = tile_df.loc[tile_df['image_id'].isin(fils)]

new_file['path'] = new_file['image_id'].apply(lambda x: '{0}{1}.jpg'.format(dir_to_use1, x))
new_file['cell_type'] = new_file['dx'].map(lesion_type_dict.get) 
new_file['cell_type_idx'] = pd.Categorical(new_file['cell_type']).codes
new_file['image'] = new_file['path'].map(lambda x: color.rgb2gray(np.asarray(Image.open(x))))

input_shape = new_file['image'].map(lambda x: x.shape).value_counts()

y = new_file.cell_type_idx

images_avg = [[[0 for _ in range(75)] for _ in range(100)] for _ in range(7)]

for im in new_file.index:
    
    cka = new_file['dx'][im]
   
    imagen = new_file['image'][im]
    
    if cka == 'nv':
        imag_class = images_avg[0]
        imag_class += imagen
        images_avg[0] = imag_class
    elif cka == 'mel':
        imag_class = images_avg[1]
        imag_class += imagen
        images_avg[1] = imag_class
    elif cka == 'bkl':
        imag_class = images_avg[2]
        imag_class += imagen
        images_avg[2] = imag_class
    elif cka == 'bec':
        imag_class = images_avg[3]
        imag_class += imagen
        images_avg[3] = imag_class
    elif cka == 'akiec':
        imag_class = images_avg[4]
        imag_class += imagen
        images_avg[4] = imag_class
    elif cka == 'vasc':
        imag_class = images_avg[5]
        imag_class += imagen
        images_avg[5] = imag_class
    elif cka == 'df':
        imag_class = images_avg[6]
        imag_class += imagen
        images_avg[6] = imag_class
                
im_type = 0
for val_c in new_file['dx'].value_counts():
    images_avg[im_type] = list(map(lambda x: np.divide(x, val_c), images_avg[im_type]))
    im_type += 1

avg_ims = [0 for _ in range(7)]

itx = 0
for im_x in images_avg:
    avg_ims[itx] = np.mean(images_avg[itx])
    itx+=1

for im in new_file.index:
    
    cka = new_file['dx'][im]
    imagen = list(new_file['image'][im].flatten())
    
    if cka == 'nv':
        imagen = list(map(lambda x: 0 if x < avg_ims[0] else 1, imagen))
    elif cka == 'mel':
        imagen = list(map(lambda x: 0 if x < avg_ims[1] else 1, imagen))
    elif cka == 'bkl':
        imagen = list(map(lambda x: 0 if x < avg_ims[2] else 1, imagen))
    elif cka == 'bec':
        imagen = list(map(lambda x: 0 if x < avg_ims[3] else 1, imagen))
    elif cka == 'akiec':
        imagen = list(map(lambda x: 0 if x < avg_ims[4] else 1, imagen))
    elif cka == 'vasc':
        imagen = list(map(lambda x: 0 if x < avg_ims[5] else 1, imagen))
    elif cka == 'df':
        imagen = list(map(lambda x: 0 if x < avg_ims[6] else 1, imagen))
        
    new_file['image'][im] = imagen
    
poolG_bool = new_file['dx'] == 'df'
poolG = list(new_file[poolG_bool]['image'])

print(len(poolG))

top_G_gn = GenAlgs(poolG, avg_ims[6])
top_G_gn = top_G_gn.get_done()
    
# Now apply genetical algorithms
poolA_bool = new_file['dx'] == 'nv'
poolA = list(new_file[poolA_bool]['image'])

top_A_gn = GenAlgs(poolA, avg_ims[0])
top_A_gn = top_A_gn.get_done()

poolB_bool = new_file['dx'] == 'mel'
poolB = list(new_file[poolB_bool]['image'])

top_B_gn = GenAlgs(poolB, avg_ims[1])
top_B_gn = top_B_gn.get_done()

poolC_bool = new_file['dx'] == 'bkl'
poolC = list(new_file[poolC_bool]['image'])

top_C_gn = GenAlgs(poolC, avg_ims[2])
top_C_gn = top_C_gn.get_done()

poolD_bool = new_file['dx'] == 'bec'
poolD = list(new_file[poolD_bool]['image'])

top_D_gn = GenAlgs(poolD, avg_ims[3])
top_D_gn = top_D_gn.get_done()

poolE_bool = new_file['dx'] == 'akiec'
poolE = list(new_file[poolE_bool]['image'])

top_E_gn = GenAlgs(poolE, avg_ims[4])
top_E_gn = top_E_gn.get_done()

poolF_bool = new_file['dx'] == 'basc'
poolF = list(new_file[poolF_bool]['image'])

top_F_gn = GenAlgs(poolF, avg_ims[5])
top_F_gn = top_F_gn.get_done()



print(top_A_gn)








    
