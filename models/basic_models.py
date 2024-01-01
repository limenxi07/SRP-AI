from fastai.vision.all import *
import os

path = '../images/'
test = '../images/A_test_set/'
reg_files = get_image_files('../images/B_training_set')
more_files = get_image_files('../images/C_expanded_training_set')

def label_func(f): return f[0].isupper()
reg_dls = ImageDataLoaders.from_name_func(path, reg_files, label_func, item_tfms=Resize(128))
more_dls = ImageDataLoaders.from_name_func(path, more_files, label_func, item_tfms=Resize(128))
print('...loaded dataloaders')

reg_learn = vision_learner(reg_dls, resnet34, metrics=accuracy)
reg_learn.fine_tune(5)
more_learn = vision_learner(more_dls, resnet34, metrics=accuracy)
more_learn.fine_tune(5)

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

test_accuracy(reg_learn, 'small dataset of real images')
test_accuracy(more_learn, 'large dataset of real images')