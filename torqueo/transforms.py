import torch
import torch.nn.functional as F

from .base import WarpTransform

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
