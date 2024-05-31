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

