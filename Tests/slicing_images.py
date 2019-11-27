#from dask.distributed import Client

#import dask_image.ndfilters
#import dask_image.ndmeasure
#import numpy as np
#import time

#import matplotlib.pyplot as plt
#from skimage import data, io
#from PIL import Image



import os
#import image_slicer
import dask.array as da
import dask_image.imread


current_directory = os.getcwd()
current_directory_images = current_directory + '/images'
images = os.listdir(current_directory_images)

current_image = dask_image.imread.imread("astronault.jpg")
print(current_image)

'''
for image in images:
  path_image = current_directory_images + '/' + image
  os.remove(path_image)




tiles = image_slicer.slice('astronault.jpg',2, save=False)
image_slicer.save_tiles(tiles, directory = current_directory_images, prefix='image', format='png')

images = os.listdir(current_directory_images)

for image in images:
  print(image)
  path_image = current_directory_images + '/' + image
  current_image = dask_image.imread.imread(current_directory_images + '/' + image)
  lines = current_image.shape[1]
  colums = current_image.shape[2]
  print(current_image)
  current_image = current_image.rechunk({1: lines/2, 2: colums/2})
  print("RECHUNK")
  print(current_image)
'''

  
