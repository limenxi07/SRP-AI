import argparse
import os

from fastai.vision.all import *

parser = argparse.ArgumentParser(description="Model Params")
parser.add_argument("--img_dir", "-i", type=str, required=True)
parser.add_argument("--model_name", "-n", type=str, required=True)
opt = parser.parse_args()

path = '../images/'
test = '../images/A_test_set/'
files = get_image_files(opt.img_dir)
def label_func(f): return f[0].isupper()
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(128))
print('...loaded dataloader')

learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(5)

def test_accuracy(learner):
  results = 0
  healthy = 0

  for image in os.scandir(test):
    if str(image)[11].isupper() and learner.predict(test + str(image)[11:-2])[0] == 'True':
      results += 1
      healthy += 1
    elif str(image)[11].islower() and learner.predict(test + str(image)[11:-2])[0] == 'False':
      results += 1

  print(f'----- Results for {opt.model_name} -----')
  print(f'Overall accuracy: {results}%')
  print(f'Healthy accuracy: {healthy * 2}%')
  print(f'Rusted accuracy: {(results - healthy) * 2}%')

test_accuracy(learn)