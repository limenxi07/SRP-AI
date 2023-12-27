from PIL import Image
import os

path = 'images/B_training_set/'
destination = 'images/E_square_training_set/'

for image in os.listdir(path):
  im = Image.open(path + image)
  im = im.resize((128, 128))
  im.save(destination + image)