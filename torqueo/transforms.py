import torch
import torch.nn.functional as F

from .base import WarpTransform

'''
Assumptions: 

1. The current version of the code assumes "centrality", i.e., the object of interest
is at the center of the image. A more accurate version would be one that uses 
ImageNet bounding boxes or clickMe maps to determine the center for warping on per image basis. 

'''

class Fisheye(WarpTransform):
    def __init__(self, strength=0.5):
        super(Fisheye, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        r_new = r ** (1 + self.strength)
        x_new = r_new * torch.cos(theta)
        y_new = r_new * torch.sin(theta)

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class BarrelDistortion(WarpTransform):
    """
    Implemented based on the single-term division model that is
    "sufficient" for most cameras but not "complete" warping equation. 
    https://en.wikipedia.org/wiki/Distortion_(optics)
    """
    def __init__(self, strength=0.5):
        super(BarrelDistortion, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)

        k = -1*self.strength
        distortion = (1-torch.sqrt(1-4*k*r**2))/(2*k*r**2)
        x_new = x_indices*distortion
        y_new = y_indices*distortion

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class PincushionDistortion(WarpTransform):
    """
    Implemented based on the single-term division model that is
    "sufficient" for most cameras but not "complete" warping equation. 
    https://en.wikipedia.org/wiki/Distortion_(optics)
    """
    def __init__(self, strength=0.5):
        super(PincushionDistortion, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)

        k = +1*self.strength
        distortion = (1-torch.sqrt(1-4*k*r**2))/(2*k*r**2)
        x_new = x_indices*distortion
        y_new = y_indices*distortion

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Compression(WarpTransform):
    '''
    Compression/Squishing is done along width only, it's simple to extend to height but not sure if that's even desired. 
    '''
    def __init__(self, strength=2):
        super(Compression, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        x_new = x_indices*self.strength 
        y_new = y_indices

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Stretching(WarpTransform):
    '''
    Stretching is along the height dimension, not the width one. Can extend if wanted. 
    '''
    def __init__(self, strength=2):
        super(Stretching, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        x_new = x_indices
        y_new = y_indices/self.strength

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Twirl(WarpTransform):
    '''
    Swirl and Twirl seem the same.
    Implemented similarly to the algo presented here: 
    https://stackoverflow.com/questions/30448045/how-do-you-add-a-swirl-to-an-image-image-distortion 
    '''
    def __init__(self, strength=3):
        super(Twirl, self).__init__()
        self.strength = strength #the number of swirl twists

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        swirl_amount = 1.0 - r/torch.sqrt(2) #most swirl at center and decreasing towards corners of image
        swirl_amount = swirl_amount.clamp(0)
        twist_angle = self.strength * swirl_amount
        x_new = r * torch.cos(theta + twist_angle)
        y_new = r * torch.sin(theta + twist_angle)

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Wave(WarpTransform):
    '''
    Warps along height and width with same sinusoidal wave.
    '''
    def __init__(self, strength=2):
        super(Wave, self).__init__()
        self.strength = strength 

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        amplitude = 0.1 * self.strength
        angular_frequency = 2 * 3.14 * 1 * self.strength  #2*pi*f*strength
        x_new = x_indices + amplitude*torch.sin(angular_frequency*x_indices)
        y_new = y_indices + amplitude*torch.sin(angular_frequency*y_indices)

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class PerspectiveWarp(WarpTransform):
    '''
    Code from RandomPerspective and perspective in PyTorch. 
    Original code: https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomPerspective 
    https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#perspective 
    '''
    def __init__(self, strength=0.5):
        super(PerspectiveWarp, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        distortion_scale = self.strength
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        
        a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float64)
        for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
            a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        b_matrix = torch.tensor(startpoints, dtype=torch.float64).view(8)
        res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution.to(torch.float32).tolist()
        #res is (a, b, c, d, e, f, g, h)
        #(x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )
        x_new = (res[0]*x_indices + res[1]*y_indices + res[2])/(res[6]*x_indices + res[7]*y_indices + 1)
        y_new = (res[3]*x_indices + res[4]*y_indices + res[5])/(res[6]*x_indices + res[7]*y_indices + 1)
        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Shear(WarpTransform):
    '''
    Equal shear in both the x and the y dimensions. 
    Implemented as described here (composition): https://en.wikipedia.org/wiki/Shear_mapping
    This is by default right leaning or top leaning shear, use negative values for the opposite effect. 
    '''
    def __init__(self, strength=0.5):
        super(Shear, self).__init__()
        self.strength = strength #cotangent of the shear angle

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        x_new = (1 + self.strength**2)*x_indices + self.strength*y_indices
        y_new = self.strength*x_indices + y_indices

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class PolarCoordinates(WarpTransform):
    def __init__(self, strength=0.5):
        super(PolarCoordinates, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        r_normalized = (r/torch.max(r))*2 - 1
        theta_normalized = (theta + 3.14)/(3.14) - 1
        x_new = theta_normalized
        y_new = r_normalized

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Pinch(WarpTransform):
    def __init__(self, strength=0.5):
        super(Pinch, self).__init__()
        assert 0.0 <= strength <= 1.0
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)

        pinch_factor = torch.sin((3.14/2)* r) ** (-1*self.strength)
        x_new = x_indices*pinch_factor
        y_new = y_indices*pinch_factor

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid

class Punch(Pinch):
    def __init__(self, strength=0.5):
        super(Punch, self).__init__(-1*strength)

class Mosaic(WarpTransform):
    '''
    Incomplete. 
    Hexagonal or other tile shape with no color averaging in the tiles. 
    The current code is using continuous warps, mosaic (discrete) isn't straighforward addition to it. 
    '''
    def __init__(self, strength=0.5):
        super(Mosaic, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        x_new = x_indices 
        y_new = y_indices

        x_new = x_new.clamp(-1, 1)
        y_new = y_new.clamp(-1, 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid
