from training_pipeline import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.data.datasets import DatasetOutput
from torchvision.io import read_image
from collections import OrderedDict
from typing import Tuple
import torchvision.transforms as transforms
import torch
import pandas as pd
import os

# custom
path = '../images/B_training_set/'
output = 'vae-model/'
annotations = '../images/healthy.csv'
os.makedirs(output, exist_ok=True)

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return DatasetOutput(data=img_path, labels=label)

# Set up the training configuration
my_training_config = BaseTrainerConfig(
  output_dir=output,
  num_epochs=200,
  learning_rate=1e-3,
  per_device_train_batch_size=200,
  per_device_eval_batch_size=200,
  train_dataloader_num_workers=2,
  eval_dataloader_num_workers=2,
  steps_saving=20,
  optimizer_cls="AdamW",
  optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
  scheduler_cls="ReduceLROnPlateau",
  scheduler_params={"patience": 5, "factor": 0.5}
)
# Set up the model configuration 
my_vae_config = model_config = VAEConfig(
  input_dim=(3, 128, 128),
  latent_dim=10
)
# Build the model
my_vae_model = VAE(
  model_config=my_vae_config
)
# Build the Pipeline
pipeline = TrainingPipeline(
	training_config=my_training_config,
	model=my_vae_model
)

# Launch the Pipeline
pipeline(
  train_data=TrainingDataset(annotations_file=annotations, img_dir=path, transform=transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), ]),
            ), # must be torch.Tensor, np.array or torch datasets
  eval_data=TrainingDataset(annotations_file=annotations, img_dir=path, transform=transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), ]), 
            ), # must be torch.Tensor, np.array or torch datasets
)