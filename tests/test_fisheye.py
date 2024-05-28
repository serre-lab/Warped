import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from torqueo import Fisheye
from torqueo import show

def test_fisheye():
    # Load image and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = Image.open("./assets/rabbit.jpg")
    img_tensor = transform(img).unsqueeze(0)

    fisheye = Fisheye(strength=0.5)
    fisheye_img = fisheye(img_tensor)

    fisheye_img_pil = transforms.ToPILImage()(fisheye_img.squeeze(0))

    show(fisheye_img_pil)
    plt.tight_layout()
    plt.savefig('assets/transformations/fisheye.jpg')
    plt.clf()
    plt.close()

    grid = fisheye.generate_warp_field(256, 256)
    fisheye.visualize_warp_field(grid)
    plt.tight_layout()
    plt.savefig('assets/warp_fields/fisheye.jpg')
    plt.clf()
    plt.close()


