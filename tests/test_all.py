import torchvision.transforms as transforms
from PIL import Image
import inspect

from torqueo.transforms import *


def find_transforms(module):
    # Find all classes in the module that inherit from WarpTransform
    transforms = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, WarpTransform) and obj is not WarpTransform:
            transforms.append(obj)
    return transforms


def test_all_transforms():
    import torqueo.transforms as transforms_module

    transform_classes = find_transforms(transforms_module)
    for transform_class in transform_classes:
        # Load image and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img = Image.open("./assets/rabbit.jpg")
        img_tensor = transform(img).unsqueeze(0)

        # Instantiate the transform with default parameters
        transform_instance = transform_class()
        warped_img = transform_instance(img_tensor)

        assert warped_img.shape == img_tensor.shape
