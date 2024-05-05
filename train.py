import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from nets import init_models, init_optimizers
from torchvision import datasets
from torch.utils.data import DataLoader
from utils import transform, generate_normal_noise, batch_to_images, show_images
from config import r_mean, r_std, dtype, device, DCGAN_CONFIG, preload_log, pretrained_discriminator, pretrained_generator, log_freq
import json


torch.set_default_dtype(dtype)


def train_model_optimizer(discriminator, generator,optimizer_generator, optimizer_discriminator,  generator_: bool, discriminator_: bool, z_samples, x_samples=None):


  assert not all([generator_, discriminator_]) and generator_ | discriminator_

  bce = nn.BCELoss()

  if discriminator_:

    if not isinstance(x_samples, torch.Tensor):

      raise ValueError("Provide x_samples to update the discriminator")

    optimizer_discriminator.zero_grad()
    zeros = torch.zeros((x_samples.shape[0],1), device=device)
    ones = torch.ones((x_samples.shape[0],1), device=device)

    loss_fake = bce(discriminator(generator(z_samples)).squeeze((2, 3)), zeros)
    loss_real = bce(discriminator(x_samples).squeeze((2, 3)), ones)

    loss_fake.backward()
    loss_real.backward()
    optimizer_discriminator.step()

    return loss_real.item(), loss_fake.item()

  if generator_:

    optimizer_generator.zero_grad()
    ones = torch.ones((z_samples.shape[0],1), device=device)

    loss_gen = bce(discriminator(generator(z_samples)).squeeze((2, 3)), ones)
    loss_gen.backward()


    optimizer_generator.step()

    return loss_gen.item(), -1


def train_loop(images_path, epochs: int, pretrained_discriminator=None, pretrained_generator=None, loss_path=None):

    # directory path parent dir / images / files 

    images = datasets.ImageFolder(images_path, transform=transform)
    train_dl = DataLoader(images, batch_size = DCGAN_CONFIG["batch_size"], shuffle=True)

    fixed_noise = generate_normal_noise(DCGAN_CONFIG["display_batch_size"], DCGAN_CONFIG["latent_dim"]).view(DCGAN_CONFIG["display_batch_size"], DCGAN_CONFIG["latent_dim"], 1, 1)
    generator, discriminator, start_epoch, step = init_models(DCGAN_CONFIG["num_channels"], bias=False, device=device, pretrained_discriminator=pretrained_discriminator ,pretrained_generator=pretrained_generator)
    optimizer_gen, optimizer_disc = init_optimizers(generator=generator, discriminator=discriminator, pretrained_discriminator=pretrained_discriminator, pretrained_generator=pretrained_generator)
    loss_real_data = 0.
    loss_fake_data = 0.
    loss_generator = 0.
    step = 0
    prev_iteration = 0


    Loss = {"loss_real_data": {}, "loss_fake_data": {}, "loss_generator": {}}

    if loss_path:
        Loss = json.load(open(loss_path, 'r'))

    for epoch in range(start_epoch, start_epoch + epochs):

        train_iterator = tqdm(train_dl)

        for iteration, (batch, _) in tqdm(enumerate(train_dl)):

            loss_fake, loss_real, loss_gen= 0., 0., 0.


            for k in range(DCGAN_CONFIG["discriminator_steps"]):
                #minibatch sampled from pg
                mpz = generate_normal_noise(len(batch), DCGAN_CONFIG["latent_dim"]).view(len(batch), DCGAN_CONFIG["latent_dim"], 1, 1)
                # train discriminator
                loss_real_data_, loss_fake_data_ = train_model_optimizer(discriminator, generator, optimizer_generator=optimizer_gen, optimizer_discriminator=optimizer_disc ,generator_ = False, discriminator_ = True, z_samples = mpz, x_samples=batch.to(device))

                loss_fake += loss_fake_data_
                loss_real += loss_real_data_


            loss_fake /= len(batch)*DCGAN_CONFIG["discriminator_steps"]
            loss_real /= len(batch)*DCGAN_CONFIG["discriminator_steps"]

            # train the generator | by descending its stochastic gradient
            mpz = generate_normal_noise(len(batch), DCGAN_CONFIG["latent_dim"]).view(len(batch), DCGAN_CONFIG["latent_dim"], 1, 1)
            loss_generator_, _ = train_model_optimizer(discriminator, generator, optimizer_generator=optimizer_gen, optimizer_discriminator=optimizer_disc ,generator_ = True, discriminator_ = False, z_samples = mpz, x_samples=None)


            loss_gen += loss_generator_ / len(batch)


            loss_real_data += loss_real
            loss_fake_data += loss_fake
            loss_generator += loss_gen

            Loss["loss_real_data"][prev_iteration + 1] = loss_real_data
            Loss["loss_fake_data"][prev_iteration + 1] = loss_fake_data
            Loss["loss_generator"][prev_iteration + 1] = loss_generator
            prev_iteration += 1

            if (iteration) % log_freq == 0:

                generated_images = generator(fixed_noise[0:DCGAN_CONFIG["display_batch_size"],:,:,:]).detach()
                ims = batch_to_images(generated_images, mean=r_mean[0], std=r_std[0], deprocess=True)
                show_images(DCGAN_CONFIG["display_batch_size"], ims, title=f"Generator Samples - Epoch {epoch + 1}")

                logger.info(f"Epoch {epoch + 1} | | -- Loss Fake data {loss_fake_data:.8f}")
                logger.info(f"Epoch {epoch + 1} | | -- Loss Real data {loss_real_data:.8f}")
                logger.info(f"Epoch {epoch + 1} | | -- Loss Generator {loss_generator:.8f}")

        if pretrained_discriminator:

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": discriminator.state_dict(),
                    "optimizer_state_dict": optimizer_disc.state_dict(),
                    "step": step
                },
                pretrained_discriminator
            )

        if pretrained_generator:


            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": optimizer_gen.state_dict(),
                    "step": step
                },
                pretrained_generator
                
            )

        if loss_path:
            with open(loss_path, 'w') as f:
                json.dump(Loss, f)

if __name__ == '__main__':
   
   from config import celeba_path, epochs, pretrained_discriminator, pretrained_generator, loss_path
   train_loop(celeba_path, epochs, pretrained_discriminator=pretrained_discriminator, pretrained_generator=pretrained_generator, loss_path=loss_path)
