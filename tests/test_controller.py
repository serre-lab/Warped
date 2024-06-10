import torchvision.transforms as transforms
from PIL import Image
import inspect
import torch

from torqueo.base import WarpTransform
import torqueo.transforms as transforms_module
from torqueo.controller import WarpController


def find_transforms(module):
    # Find all classes in the module that inherit from WarpTransform
    transforms = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, WarpTransform) and obj is not WarpTransform:
            transforms.append(obj)
    return transforms

def test_identity_transform():
    # test that distortion level 0 causes no transformation in the image for all warps
    transform_classes = find_transforms(transforms_module)
    for transform_class in transform_classes:
        # Load image and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img = Image.open("./assets/rabbit.jpg")
        img_tensor = transform(img).unsqueeze(0)

        # Controller for the transform class
        transform_controller = WarpController(transform_class)
        # Instantiate the transform with 0 distortion level for identity transform
        transform_instance = transform_controller(0)
        warped_img = transform_instance(img_tensor)

        assert torch.allclose(warped_img,img_tensor, rtol=3e-05), \
            f"Class {transform_class}, warped shape{warped_img.shape}, original shape{img_tensor.shape}"

def test_same_distortion_levels_different_range():
    # Test controller for same distortion levels but different strength ranges
    transform_classes = find_transforms(transforms_module)
    for transform_class in transform_classes:
        # Load image and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img = Image.open("./assets/rabbit.jpg")
        img_tensor = transform(img).unsqueeze(0)

        class_name = transform_class.__name__
        # The 2nd strength range is double the first one
        range_2 = (0, 1)
        range_double = (range_2[0], range_2[0] + 2*(range_2[1]-range_2[0]))
        if class_name == "Implosion":
            range_2 = (0.5, 1)
            range_double = (range_2[1] - 2*(range_2[1]-range_2[0]), range_2[1])

        # Two controllers with both having same max distortion level
        #(s0, s1, s2)
        level = 2
        controller = WarpController(
            transform_class, max_distortion_level = level, strength_range = range_2
        )
        #(s0, s0 + (s2-s0), s0 + 2*(s2-s0)), where (s2-s0)=(s1-s0)*2
        controller_double = WarpController(
            transform_class, max_distortion_level = level, strength_range = range_double
        )

        transformer = controller(1)
        transformer_double = controller_double(1)

        # Check that strength is as expected.
        if class_name == "Implosion":
            assert range_2[1] - (range_2[1] - transformer.strength)*level == transformer_double.strength
        else:
            assert range_2[0] + (transformer.strength - range_2[0])*level == transformer_double.strength

def test_different_distortion_levels_same_range():
    # Controllers with different distortion levels but same strength range and same output.
    transform_classes = find_transforms(transforms_module)
    for transform_class in transform_classes:
        # Load image and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img = Image.open("./assets/rabbit.jpg")
        img_tensor = transform(img).unsqueeze(0)

        # Two controller with 5 and 10 max distortion level and the same range of strength
        #s0, s1, s2, s3, s4, s5
        controller_5 = WarpController(transform_class, max_distortion_level=5)
        transformer_5 = controller_5(2)
        #s0, 0.5(s1-s0) + s0, s1, 0.5(s2-s1) + s1, s2, ...
        controller_10 = WarpController(transform_class, max_distortion_level=10)
        transformer_10 = controller_10(4)

        # Strengths are equal (s2) as they are sampled from evenly spaced same strength range
        assert transformer_5.strength == transformer_10.strength

        warped_img_5 = transformer_5(img_tensor)
        warped_img_10 = transformer_10(img_tensor)

        # The warped images for same strength should be same
        if transform_class.__name__=='Perspective':
            # random transform can't be compared
            continue
        assert torch.allclose(warped_img_5, warped_img_10), \
            f"Class {transform_class}, shape 5:{warped_img_5.shape}, 10:{warped_img_10.shape}"

def test_different_distortion_levels_different_range():
    # Test controller for different distortion levels and different strength ranges but same output
    transform_classes = find_transforms(transforms_module)
    for transform_class in transform_classes:
        # Load image and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        img = Image.open("./assets/rabbit.jpg")
        img_tensor = transform(img).unsqueeze(0)

        class_name = transform_class.__name__
        # The 2nd strength range is double the first one
        range_2 = (0, 1)
        range_4 = (range_2[0], range_2[0] + 2*(range_2[1]-range_2[0]))
        if class_name == "Implosion":
            range_2 = (0.5, 1)
            range_4 = (range_2[1] - 2*(range_2[1]-range_2[0]), range_2[1])

        # Two controllers with 2 & 4 max distortion levels
        #(s0, s1, s2)
        controller_2 = WarpController(transform_class,max_distortion_level=2,strength_range=range_2)
        transformer_2 = controller_2(1)
        #(s0, s1, s2, s3, s4)
        controller_4 = WarpController(transform_class,max_distortion_level=4,strength_range=range_4)
        transformer_4 = controller_4(1)

        # Both strength should be same irrespective of having different levels and ranges
        assert transformer_2.strength == transformer_4.strength

        warped_img_2 = transformer_2(img_tensor)
        warped_img_4 = transformer_4(img_tensor)

        # The warped images for same strength should be same
        if transform_class.__name__=='Perspective':
            # random transform can't be compared
            continue
        assert torch.allclose(warped_img_2, warped_img_4), \
            f"Class {transform_class}, shape 2:{warped_img_2.shape}, 4:{warped_img_4.shape}"
