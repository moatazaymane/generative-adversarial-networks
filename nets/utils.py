import torch
import torch.nn as nn
from config import dtype, DCGAN_CONFIG, image_size as img_size, num_channels

def make_layer_dcgan_generator(layer_num:int, in_features: int, out_features: int,kernel_size, stride, padding, relu = True, dropout=None, BN=False, bias=False, dtype=dtype):

  layers = []
  layers.append([f'ConvT{layer_num}', nn.ConvTranspose2d(in_channels = in_features, out_channels = out_features, kernel_size=kernel_size, stride = stride, padding=padding, bias=bias, dtype= dtype)])

  if BN:

    layers.append([f"Batch_Norm{layer_num}", nn.BatchNorm2d(out_features, dtype=dtype)])


  if relu:
    layers.append([f"Relu{layer_num}", nn.ReLU()])

  return layers

def make_layer_dcgan_discriminator(layer_num:int, in_features: int, out_features: int,kernel_size, stride, padding, lrelu = True, dropout=None, BN=False, bias=False, dtype=dtype):



  layers = []
  layers.append([f'Conv{layer_num}', nn.Conv2d(in_channels = in_features, out_channels = out_features, kernel_size=kernel_size, stride = stride,padding=padding, bias=bias, dtype= dtype)])

  if BN:

    layers.append([f"Batch_Norm{layer_num}", nn.BatchNorm2d(out_features, dtype=dtype)])


  if lrelu:
    layers.append([f"Leaky_Relu{layer_num}", nn.LeakyReLU(DCGAN_CONFIG["leaky_relu"])])

  return layers


def make_layer(layer_num:int, in_features: int, out_features: int, relu = True, dropout=None, BN=False, dtype=dtype):

  layers = []
  layers.append([f'Linear{layer_num}', nn.Linear(in_features = in_features, out_features=out_features, dtype=dtype)])

  if BN:

    layers.append([f"Batch_Norm{layer_num}", nn.BatchNorm1d(out_features, dtype=dtype)])

  if dropout:

    if type(dropout) != float:
      raise ValueError('Dropout value should be float')
    layers.append([f"Dropout{layer_num}",nn.Dropout(p=dropout)])

  if relu:
    layers.append([f"Leaky_Relu{layer_num}", nn.LeakyReLU(DCGAN_CONFIG['leaky_relu'])])

  return layers