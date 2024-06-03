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
        assert img.dim() == 4

        batch_size, channels, height, width = img.size()
        grid = self.generate_warp_field(height, width)
        grid = grid.to(img.device)

        warped_img = F.grid_sample(img, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        return warped_img

    def visualize_warp_field(self, height=1024, width=1024, padding=1, bg_color=1.0, grid_color=.13):
        # Create an intermediate white image with an 8x8 grid
        intermediate_img = torch.ones(1, height, width) * bg_color

        step_y = height // 8
        step_x = width // 8

        for i in range(9):
          y = i * step_y
          y_min = max(0, y - padding)
          y_max = min(height, y + padding)
          intermediate_img[:, y_min:y_max, :] = grid_color

          x = i * step_x
          x_min = max(0, x - padding)
          x_max = min(width, x + padding)
          intermediate_img[:, :, x_min:x_max] = grid_color

        warped_img = self.forward(intermediate_img.unsqueeze(0))[0]

        plt.subplot(1, 2, 1)
        show(intermediate_img, cmap="gray", vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        show(warped_img, cmap="gray", vmin=0, vmax=1)
