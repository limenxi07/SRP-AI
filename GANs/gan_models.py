from fastai.vision.all import *
import os

path = '../results'
test = '../images/A_test_set'
gan_files = get_image_files('../results/gan_results')
lsgan_files = get_image_files('../results/lsgan_results')
wgan_files = get_image_files('../results/wgan_results')

def label_func(f): return f[0].isupper()
gan_dls = ImageDataLoaders.from_name_func(path, gan_files, label_func, item_tfms=Resize(128))
lsgan_dls = ImageDataLoaders.from_name_func(path, lsgan_files, label_func, item_tfms=Resize(128))
wgan_dls = ImageDataLoaders.from_name_func(path, wgan_files, label_func, item_tfms=Resize(128))
print('loaded dataloaders')

gan_learn = vision_learner(gan_dls, resnet34, metrics=accuracy)
gan_learn.fine_tune(5)
print('trained gan model')

lsgan_learn = vision_learner(lsgan_dls, resnet34, metrics=accuracy)
lsgan_learn.fine_tune(5)
print('trained lsgan model')

wgan_learn = vision_learner(wgan_dls, resnet34, metrics=accuracy)
wgan_learn.fine_tune(5)
print('trained wgan model')

def test_accuracy(learner, model):
  results = 0
  healthy = 0

  print('starting tests for ' + model)

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

test_accuracy(gan_learn, 'gan')
test_accuracy(lsgan_learn, 'lsgan')
test_accuracy(wgan_learn, 'wgan')