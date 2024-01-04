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

# CUSTOM DATASET
mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:10000].reshape(-1, 1, 28, 28) / 255.
train_targets = mnist_trainset.targets[:10000]
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
eval_targets = mnist_trainset.targets[-10000:]

if not os.path.exists("data_folders"):
    os.mkdir("data_folders")
if not os.path.exists("data_folders/train"):
    os.mkdir("data_folders/train")
if not os.path.exists("data_folders/eval"):
    os.mkdir("data_folders/eval")

for i in range(len(train_dataset)):
    img = 255.0*train_dataset[i][0].unsqueeze(-1)
    img_folder = os.path.join("data_folders", "train", f"{train_targets[i]}")
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    imageio.imwrite(os.path.join(img_folder, "%08d.jpg" % i), np.repeat(img, repeats=3, axis=-1).type(torch.uint8))

for i in range(len(eval_dataset)):
    img = 255.0*eval_dataset[i][0].unsqueeze(-1)
    img_folder = os.path.join("data_folders", "eval", f"{eval_targets[i]}")
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    imageio.imwrite(os.path.join(img_folder, "%08d.jpg" % i), np.repeat(img, repeats=3, axis=-1).type(torch.uint8))

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
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
    latent_dim=16
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

pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset
)
last_training = sorted(os.listdir(output))[-1]
trained_model = AutoModel.load_from_folder(os.path.join(output, last_training, 'final_model'))
# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)
# sample
gen_data = normal_samper.sample(
    num_samples=25
)
# show results with normal sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
# set up GMM sampler config
gmm_sampler_config = GaussianMixtureSamplerConfig(
    n_components=10
)

# create gmm sampler
gmm_sampler = GaussianMixtureSampler(
    sampler_config=gmm_sampler_config,
    model=trained_model
)

# fit the sampler
gmm_sampler.fit(train_dataset)
# sample
gen_data = gmm_sampler.sample(
    num_samples=25
)
# show results with gmm sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('results.jpg')
reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('reconstructions.jpg')
# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(eval_dataset[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('true_data.jpg')
interpolations = trained_model.interpolate(eval_dataset[:5].to(device), eval_dataset[5:10].to(device), granularity=10).detach().cpu()
# show interpolations
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))

for i in range(5):
    for j in range(10):
        axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.savefig('interpolations.jpg')