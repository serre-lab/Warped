import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from torqueo import Perspective
from torqueo import show


def test_perspective():
    # Load image and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = Image.open("./assets/rabbit.jpg")
    img_tensor = transform(img).unsqueeze(0)

    transform = Perspective(strength=0.5)
    warped_img = transform(img_tensor)[0]

    show(warped_img)
    plt.tight_layout()
    plt.savefig('assets/transformations/perspective.jpg', bbox_inches='tight')
    plt.clf()
    plt.close()

    transform.visualize_warp_field()
    plt.tight_layout()
    plt.savefig('assets/warp_fields/perspective.jpg', bbox_inches='tight')
    plt.clf()
    plt.close()
