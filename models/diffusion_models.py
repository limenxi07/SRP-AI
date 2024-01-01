from fastai.vision.all import *
import os

path = '../results'
test = '../images/A_test_set/'
real_files = get_image_files('../results/diffusion_results')
alt_files = get_image_files('../results/diffusion_results_2')

def label_func(f): return f[0].isupper()
real_dls = ImageDataLoaders.from_name_func(path, real_files, label_func, item_tfms=Resize(128))
alt_dls = ImageDataLoaders.from_name_func(path, alt_files, label_func, item_tfms=Resize(128))
print('...loaded dataloaders')

real_learn = vision_learner(real_dls, resnet34, metrics=accuracy)
real_learn.fine_tune(5)
alt_learn = vision_learner(alt_dls, resnet34, metrics=accuracy)
alt_learn.fine_tune(5)

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

test_accuracy(real_learn, 'diffusion model trained on real images')
test_accuracy(alt_learn, 'diffusion model trained on altered images')