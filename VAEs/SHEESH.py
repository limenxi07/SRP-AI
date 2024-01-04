import torch
import matplotlib.pyplot as plt
from vae import VAE
from tqdm import tqdm

import torchvision
import torchvision.datasets as datasets
import os
import random
from PIL import Image
from utils import train_model, image_view, CustomDataset

torch.set_default_dtype(torch.float64)

# CUSTOM DATASET
dataset_iter = iter(torch.utils.data.DataLoader(mnist, shuffle=True))
dataset = iter()

mnist = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    mnist_iter = iter(torch.utils.data.DataLoader(mnist, shuffle=True))
    
    num_train = 60_000  # note cannot be greater than 60_000 for MNIST
    binary_pixels = False

    training_images = []
    for _ in tqdm(range(num_train)):
        img = next(mnist_iter)[0].squeeze(0).permute(1, 2, 0) # swap 'mnist_iter' for 'celeba_iter'
        if binary_pixels:
            img = img.round()
        training_images.append(img)