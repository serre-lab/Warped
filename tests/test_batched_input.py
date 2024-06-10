import torchvision.transforms as transforms
from PIL import Image
import inspect
import torch

from torqueo.base import WarpTransform

class IdentityTransform(WarpTransform):
    def __init__(self):
        super().__init__()

    def generate_warp_field(self, height, width):
        # do no transformation, output image = input image
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))
        x_new = x_indices
        y_new = y_indices
        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

def test_for_batched_input():
    # Check that the base class, i.e., WarpTransform supports batched images
    # Load image and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img1 = transform(Image.open("./assets/rabbit.jpg"))
    img2 = transform(Image.open("./assets/turtle.jpg"))
    batch = torch.stack((img1, img2), dim=0)

    # Instantiate the transform with default parameters
    transform_instance = IdentityTransform()
    try:
        warped_img = transform_instance(batch)
    except Exception as e:
        return f"Batched input test failing: {e}"

    assert warped_img.shape == batch.shape
