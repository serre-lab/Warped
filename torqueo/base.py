import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .plots import show

class WarpTransform(torch.nn.Module):
    def __init__(self):
        super(WarpTransform, self).__init__()

    def generate_warp_field(self, height, width):
        raise NotImplementedError("Subclasses must implement generate_warp_field")

    def forward(self, img):
        batch_size, channels, height, width = img.size()
        grid = self.generate_warp_field(height, width)
        grid = grid.to(img.device)

        warped_img = F.grid_sample(img, grid, align_corners=True, mode='bilinear', padding_mode='border')
        return warped_img

    def visualize_warp_field(self, grid):
        for i in range(2):
            plt.subplot(1, 2, i+1)
            show(grid[0, :, :, i].cpu().numpy())
