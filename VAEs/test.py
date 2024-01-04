import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pythae.data.datasets import DatasetOutput
from pythae.models import VAE, AutoModel, VAEConfig
from pythae.models.nn.benchmarks.mnist import (Decoder_ResNet_AE_MNIST,
                                               Encoder_ResNet_VAE_MNIST)
from pythae.pipelines.training import TrainingPipeline
from pythae.samplers import (GaussianMixtureSampler,
                             GaussianMixtureSamplerConfig, NormalSampler)
from pythae.trainers import BaseTrainerConfig
from torchvision import datasets, transforms

path = 'images'
output = 'vae_healthy'

# # CUSTOM DATASET
# mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)

# train_dataset = mnist_trainset.data[:10000].reshape(-1, 1, 28, 28) / 255.
# train_targets = mnist_trainset.targets[:10000]
# eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
# eval_targets = mnist_trainset.targets[-10000:]

# for i in range(len(train_dataset)):
#     img = 255.0*train_dataset[i][0].unsqueeze(-1)
#     img_folder = os.path.join("images", "healthy", f"{train_targets[i]}")
#     if not os.path.exists(img_folder):
#         os.mkdir(img_folder)
#     imageio.imwrite(os.path.join(img_folder, "%08d.jpg" % i), np.repeat(img, repeats=3, axis=-1).type(torch.uint8))

# for i in range(len(eval_dataset)):
#     img = 255.0*eval_dataset[i][0].unsqueeze(-1)
#     img_folder = os.path.join("images", "healthy", f"{eval_targets[i]}")
#     if not os.path.exists(img_folder):
#         os.mkdir(img_folder)
#     imageio.imwrite(os.path.join(img_folder, "%08d.jpg" % i), np.repeat(img, repeats=3, axis=-1).type(torch.uint8))

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor() # the data must be tensors
])

class MyCustomDataset(datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        X, _ = super().__getitem__(index)

        return DatasetOutput(
            data=X
        )

train_dataset = MyCustomDataset(
    root=path,
    transform=data_transform,
)

eval_dataset = MyCustomDataset(
    root=path, 
    transform=data_transform
)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = BaseTrainerConfig(
    output_dir=output,
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=10, 
    optimizer_cls="AdamW",
    optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)}
)


model_config = VAEConfig(
    input_dim=(3, 128, 128),
    latent_dim=32728
)

model = VAE(
    model_config=model_config,
    encoder=Encoder_ResNet_VAE_MNIST(model_config), 
    decoder=Decoder_ResNet_AE_MNIST(model_config) 
)

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

print(model.model_config)