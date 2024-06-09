import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .plots import show


class WarpTransform(torch.nn.Module):
    """
    A base class for applying warp transformations to images using a grid sampling approach.

    This class provides the framework for implementing various warp transformations by
    defining a common interface. Subclasses should implement the `generate_warp_field`
    method to provide the specific warp grid for the transformation.
    """

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the transformation.

        Parameters
        ----------
        height : int
            The height of the image.
        width : int
            The width of the image.

        Returns
        -------
        torch.Tensor
            The grid for warping the image.
        """
        raise NotImplementedError("Subclasses must implement generate_warp_field")

    def forward(self, img):
        """
        Applies the warp transformation to the input image tensor.

        Parameters
        ----------
        img : torch.Tensor
            The input image tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            The warped image tensor.
        """
        assert img.dim() == 4

        batch_size, _, height, width = img.size()
        grid = self.generate_warp_field(height, width)
        grid = grid.to(img.device)
        grid = grid.repeat(batch_size, 1, 1, 1)

        warped_img = F.grid_sample(img, grid, align_corners=True,
                                   mode='bilinear', padding_mode='zeros')
        return warped_img

    def visualize_warp_field(
            self, height=1024, width=1024, padding=1, bg_color=1.0, grid_color=0.13):
        """
        Visualizes the warp field by creating an intermediate image with a grid and
        applying the warp transformation.

        Parameters
        ----------
        height : int, optional
            The height of the intermediate image (default is 1024).
        width : int, optional
            The width of the intermediate image (default is 1024).
        padding : int, optional
            The padding around the grid lines (default is 1).
        bg_color : float, optional
            The background color of the intermediate image (default is 1.0, white).
        grid_color : float, optional
            The color of the grid lines (default is 0.13, dark gray).
        """
        # pylint: disable=C0103

        # create a simple 8x8 grid and apply warping on it
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
