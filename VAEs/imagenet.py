import argparse
import logging
import os
import time

import hostlist
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pythae.data.datasets import DatasetOutput
from pythae.models import VQVAE, AutoModel, VQVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.nn.benchmarks.utils import ResBlock
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.samplers import PixelCNNSampler, PixelCNNSamplerConfig
from pythae.pipelines import GenerationPipeline
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
output = 'models/vae_healthy'

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="imagenet-distributed",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="pythae",
)

args = ap.parse_args()

training_config = BaseTrainerConfig(
        num_epochs=1,
        train_dataloader_num_workers=8,
        eval_dataloader_num_workers=8,
        output_dir=output,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-4,
        steps_saving=None,
        steps_predict=None,
        no_cuda=False,
        dist_backend="nccl",
    )


class Encoder_ResNet_VQVAE_ImageNet(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 4, 2, padding=1),
            nn.Conv2d(32, 64, 4, 2, padding=1),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.Conv2d(128, 128, 4, 2, padding=1),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
        )

        self.pre_qantized = nn.Conv2d(128, self.latent_dim, 1, 1)

    def forward(self, x: torch.Tensor):
        output = ModelOutput()
        out = x
        out = self.layers(out)
        output["embedding"] = self.pre_qantized(out)

        return output


class Decoder_ResNet_VQVAE_ImageNet(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)

        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.dequantize = nn.ConvTranspose2d(self.latent_dim, 128, 1, 1)

        self.layers = nn.Sequential(
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            ResBlock(in_channels=128, out_channels=64),
            nn.ConvTranspose2d(128, 128, 4, 2, padding=1),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.ConvTranspose2d(32, self.n_channels, 4, 2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        out = self.dequantize(z)
        output["reconstruction"] = self.layers(out)

        return output


class CustomDataset(Dataset):
    def __init__(self, data_dir=None, transforms=None):
        self.imgs_path = [os.path.join(data_dir, n) for n in os.listdir(data_dir)]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return DatasetOutput(data=img)
    
    def max(self):
        return len(self.imgs_path)
    
    def min(self):
        return 1


def main(args):

    img_transforms = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    train_dataset = CustomDataset(
        data_dir="images/healthy",
        transforms=img_transforms,
    )
    eval_dataset = CustomDataset(
        data_dir="images/healthy",
        transforms=img_transforms,
    )

    model_config = VQVAEConfig(
        input_dim=(3, 128, 128), latent_dim=128, use_ema=True, num_embeddings=1024
    )

    encoder = Encoder_ResNet_VQVAE_ImageNet(model_config)
    decoder = Decoder_ResNet_VQVAE_ImageNet(model_config)

    model = VQVAE(model_config=model_config, encoder=encoder, decoder=decoder)

    callbacks = []

    # Only log to wandb if main process
    if args.use_wandb and (training_config.rank == 0 or training_config == -1):
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        callbacks.append(wandb_cb)

    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        callbacks=callbacks,
    )

    start_time = time.time()

    trainer.train()

    end_time = time.time()

    logger.info(f"Total execution time: {(end_time - start_time)} seconds")


  # GENERATE IMAGES
    last_training = sorted(os.listdir(output))[-1]
    trained_model = AutoModel.load_from_folder(os.path.join(output, last_training, 'final_model'))
    sampler_config = PixelCNNSamplerConfig(
      n_layers=10
    )
    pixelcnn_sampler = PixelCNNSampler(
        sampler_config=sampler_config,
        model=trained_model
    )
    # pixelcnn_sampler.fit(train_dataset.data)
    # samples = pixelcnn_sampler.sample(num_samples=10)
    print(train_dataset)


if __name__ == "__main__":

    main(args)