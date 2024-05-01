import torch
import torch.nn as nn
from nets.utils import make_layer_dcgan_generator, make_layer_dcgan_discriminator
from collections import OrderedDict
from config import DCGAN_CONFIG
from torch.optim import Adam


class DCGAN_Generator(nn.Module):

  def __init__(self, num_channels: int, bias: bool):

    super().__init__()
    layers = []
    layers += [*make_layer_dcgan_generator(layer_num = 1, in_features = DCGAN_CONFIG["latent_dim"], out_features = 1024 ,kernel_size = 4, stride = 1, padding=0, relu = True, BN=True, bias=bias)]
    layers += [*make_layer_dcgan_generator(layer_num = 2, in_features = 1024, out_features = 512, kernel_size = 4, stride = 2, padding=1 , relu = True, BN=True,bias=bias)]
    layers += [*make_layer_dcgan_generator(layer_num = 3, in_features = 512, out_features = 256, kernel_size =4, stride = 2,padding=1, relu = True, BN=True,bias=bias)]
    layers += [*make_layer_dcgan_generator(layer_num = 4, in_features = 256, out_features = 128, kernel_size =4, stride = 2,padding=1, relu = True, BN=True,bias=bias)]
    layers += [*make_layer_dcgan_generator(layer_num = 5, in_features = 128, out_features = 64, kernel_size =4, stride = 2,padding=1, relu = True, BN=True,bias=bias)]
    layers.append(["ConvT6", nn.ConvTranspose2d(64, num_channels, 4, 2, 1, bias=False)])
    layers.append(["Tanh", nn.Tanh()])

    self.generator = nn.ModuleDict(layers)

  def forward(self, z):

    out = z
    for layer_name, layer in self.generator.items():
      out = layer(out)

    return out


class DCGAN_Discriminator(nn.Module):

  def __init__(self, num_channels: int, img_size, bias: bool):

    super().__init__()
    layers = []
    layers += [*make_layer_dcgan_discriminator(layer_num = 1, in_features = num_channels, out_features = 64 ,kernel_size = 4, stride = 2, padding=1, lrelu = True, BN=False, bias=bias)]
    layers += [*make_layer_dcgan_discriminator(layer_num = 2, in_features = 64, out_features = 128, kernel_size = 4, stride = 2, padding=1 , lrelu = True, BN=True,bias=bias)]
    layers += [*make_layer_dcgan_discriminator(layer_num = 3, in_features = 128, out_features = 256, kernel_size =4, stride = 2,padding=1, lrelu = True, BN=True,bias=bias)]
    layers += [*make_layer_dcgan_discriminator(layer_num = 4, in_features = 256, out_features = 512, kernel_size =4, stride = 2,padding=1, lrelu = True, BN=True,bias=bias)]
    layers += [*make_layer_dcgan_discriminator(layer_num = 5, in_features = 512, out_features = 256, kernel_size =4, stride = 2,padding=1, lrelu = True, BN=True,bias=bias)]

    layers.append(["ConvT6", nn.Conv2d(256, 1, 4, 1, 0, bias=False)])
    layers.append(["Leaky_Relu6", nn.LeakyReLU(DCGAN_CONFIG["leaky_relu"])])
    layers.append(["Sigmoid", nn.Sigmoid()])

    self.discriminator = nn.ModuleDict(layers)
    self.img_size = img_size

  def forward(self, z):

    out = z
    for layer_name, layer in self.discriminator.items():
      out = layer(out)

    return out

def init_models(num_channels, bias, device, pretrained_generator=None, pretrained_discriminator=None):

    generator = DCGAN_Generator(num_channels=num_channels, bias=bias).to(device)
    discriminator = DCGAN_Discriminator(num_channels=num_channels, img_size = DCGAN_CONFIG["image_size"], bias=bias).to(device)

    to_init = [generator.generator]
    step = 0
    start_epoch = 0

    if pretrained_generator:

        try:

            print('here')
            state_generator = torch.load(pretrained_generator)
            generator.load_state_dict(state_generator["model_state_dict"])
            step = state_generator['step']
            start_epoch = state_generator['epoch']
            to_init.pop()

        except Exception:
           print('Error in loading generator state -- initializing a generator')

    to_init.append(discriminator.discriminator)
    if pretrained_discriminator:

        try:
            state_discriminator = torch.load(pretrained_discriminator)
            discriminator.load_state_dict(state_discriminator["model_state_dict"])
            step = state_discriminator['step']
            to_init.pop()

        except Exception:
           print('Error in loading discriminator state -- initializing a discriminator')

    for model in to_init:
      for layer_name, layer in model.items():

        if "Conv" in layer_name:
            torch.nn.init.normal_(layer.weight.data, DCGAN_CONFIG["init_params"]["mean"], DCGAN_CONFIG["init_params"]["std"])

        elif "Batch_Norm" in layer_name:
            torch.nn.init.normal_(layer.weight.data, 1, DCGAN_CONFIG["init_params"]["std"])
            torch.nn.init.constant_(layer.bias.data, 0.)

    return generator, discriminator, start_epoch, step

def init_optimizers(generator, discriminator, pretrained_generator=None, pretrained_discriminator=None):
    optimizer_gen = Adam(params = generator.parameters(), lr= DCGAN_CONFIG["lr"], betas=DCGAN_CONFIG["betas"])
    optimizer_disc = Adam(params = discriminator.parameters(), lr= DCGAN_CONFIG["lr"], betas=DCGAN_CONFIG["betas"])

    if pretrained_discriminator:

      try:
        state_discriminator = torch.load(pretrained_discriminator)
        optimizer_disc.load_state_dict(state_discriminator["optimizer_state_dict"])
      
      except Exception:
         print("Error loading optimizer state dictionary -- Discriminator")
    
    if pretrained_generator:

      try:
        state_generator = torch.load(pretrained_generator)
        optimizer_gen.load_state_dict(state_generator["optimizer_state_dict"])
      
      except Exception:
         print("Error loading optimizer state dictionary -- Generator")


    return optimizer_gen, optimizer_disc


  