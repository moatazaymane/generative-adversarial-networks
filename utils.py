import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from torchvision.transforms import v2
from config import DCGAN_CONFIG, device, r_mean, r_std


transform = v2.Compose([v2.CenterCrop(DCGAN_CONFIG["image_size"]), v2.ToTensor(), v2.Normalize(r_mean, r_std)])

def generate_normal_noise(size: int, noise_dim: int):

  noise_images = torch.normal(mean = torch.zeros((size, noise_dim), device=device), std = torch.ones((size, noise_dim) , device=device))

  return noise_images


def deprocess_image(image_, mean, std):

  image = torch.clone(image_)
  image.mul_(std).add_(mean)
  image.mul_(255.)

  return image.to(dtype=torch.uint8)


def batch_to_images(image_batch, mean=None, std=None, deprocess=False) -> List[Image.Image]:

    if deprocess:

      deprocess_batch = deprocess_image(image_batch, mean, std).cpu().numpy()
    
    else:
        deprocess_batch = torch.clone(image_batch).cpu().numpy()


    deprocess_batch = deprocess_batch.transpose(0, 2, 3, 1)

    return [Image.fromarray(image) for image in deprocess_batch]


def show_images(num_images, images_list: List[Image.Image], images_titles = None, title=None) -> None:
    n: int = len(images_list)
    f = plt.figure(figsize=(10, 10))

    for i in range(num_images):
        # Debug, plot figure
        ax = f.add_subplot(5, 4, i + 1)
        if images_titles:
          ax.title.set_text(images_titles[i])
        plt.imshow(images_list[i])
        ax.axis('off')
      
    if title:
       plt.suptitle(title, fontsize=14)
      
    plt.show(block=True)