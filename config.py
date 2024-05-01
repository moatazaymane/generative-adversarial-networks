import torch

#config
dtype = torch.float32
torch.set_default_dtype(dtype)
#config

device = "cuda" if torch.cuda.is_available() else "cpu"

DCGAN_CONFIG = dict(image_size=(128, 128),num_channels=3,display_batch_size = 20, batch_size=300, device=device, lr= 2e-4, leaky_relu= 0.2, latent_dim = 100,  betas = (0.5, 0.999), init_params = {"mean":0, "std": 0.02}, discriminator_steps=1)

mnist_mean = torch.tensor(0.1306)
mnist_std = torch.tensor(0.308)
image_size = (128, 128)
num_channels = 3
batch_size = 300
discriminator_steps = 1
log_freq = 20 # log loss and visualize generator samples
r_mean, r_std = (.5, .5, .5), (.5, .5, .5)
celeba_path = "/Users/a__/Desktop/datasets"
epochs = 50
pretrained_discriminator=None
pretrained_generator=None
loss_path=None
preload_log = False