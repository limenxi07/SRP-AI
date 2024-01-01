import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
from fastai.vision.all import *

path = '../images/B_training_set/'
destination = '../images/D_augmented_set'

def brighten(image):
  enhancer = ImageEnhance.Brightness(image)
  return enhancer.enhance(random.randint(10, 25)/10), True

def flip(image):
  new_image = ImageOps.flip(image)
  return ImageOps.mirror(new_image), True

def rotate(image):
  return image.rotate(random.randint(1, 359)), True

def gaussian_noise(image):
  new_image = np.asarray(image)
  noise = np.random.normal(loc=1.0, scale=random.randint(100, 400)/1000, size=new_image.shape)
  new_image = new_image * noise
  new_image = new_image.astype(np.uint8).clip(0, 255)
  return new_image, False

def save(image, new_image, boolean, n):
  if boolean:
    new_image = new_image.save(f'{destination}/{image}-{n}.jpg')
  else:
    plt.imsave(f'{destination}/{image}-{n}.jpg', new_image)

for file in os.listdir(path):
  image = Image.open(path + file)
# (1) FLIP + BRIGHTNESS 
  new_image, boolean = flip(image)
  new_image, boolean = brighten(new_image)
  save(file, new_image, boolean, 1)
  # (2) FLIP + ROTATION
  new_image, boolean = flip(image)
  new_image, boolean = rotate(new_image)
  save(file, new_image, boolean, 2)
  # (3) FLIP + GAUSSIAN NOISE
  new_image, boolean = flip(image)
  new_image, boolean = gaussian_noise(new_image)
  save(file, new_image, boolean, 3)
  # (4) BRIGHTNESS + ROTATION
  new_image, boolean = brighten(image)
  new_image, boolean = rotate(new_image)
  save(file, new_image, boolean, 4)
  # (5) BRIGHTNESS + GAUSSIAN NOISE
  new_image, boolean = brighten(image)
  new_image, boolean = gaussian_noise(new_image)
  save(file, new_image, boolean, 5)
  # (6) ROTATION + GAUSSIAN NOISE
  new_image, boolean = rotate(image)
  new_image, boolean = gaussian_noise(new_image)
  save(file, new_image, boolean, 6)
  # (7) FLIP + BRIGHTNESS + ROTATION
  new_image, boolean = flip(image)
  new_image, boolean = brighten(new_image)
  new_image, boolean = rotate(new_image)
  save(file, new_image, boolean, 7)
  # (8) FLIP + ROTATION + GAUSSIAN NOISE
  new_image, boolean = flip(image)
  new_image, boolean = rotate(new_image)
  new_image, boolean = gaussian_noise(new_image)
  save(file, new_image, boolean, 8)
  # (9) FLIP + BRIGHTNESS + GAUSSIAN NOISE
  new_image, boolean = flip(image)
  new_image, boolean = brighten(new_image)
  new_image, boolean = gaussian_noise(new_image)
  save(file, new_image, boolean, 9)
  # (10) BRIGHTNESS + ROTATION + GAUSSIAN NOISE
  new_image, boolean = brighten(image)
  new_image, boolean = rotate(new_image)
  new_image, boolean = gaussian_noise(new_image)
  save(file, new_image, boolean, 10)


# AI MODEL TRAINED ON ALTERED IMAGES
path = '../images/'
test = '../images/A_test_set/'
files = get_image_files(destination)
def label_func(f): return f[0].isupper()
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(480))
print('...loaded dataloader')

learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(5)

def test_accuracy(learner, model):
  results = 0
  healthy = 0

  for image in os.scandir(test):
    if str(image)[11].isupper() and learner.predict(test + str(image)[11:-2])[0] == 'True':
      results += 1
      healthy += 1
    elif str(image)[11].islower() and learner.predict(test + str(image)[11:-2])[0] == 'False':
      results += 1

  print(f'----- Results for {model} -----')
  print(f'Overall accuracy: {results}%')
  print(f'Healthy accuracy: {healthy * 2}%')
  print(f'Rusted accuracy: {(results - healthy) * 2}%')

test_accuracy(learn, 'altered images')