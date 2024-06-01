import torch
import torch.nn.functional as F

from .base import WarpTransform

class Fisheye(WarpTransform):
    """
    Applies a fisheye distortion effect by stretching the magnitude of the coordinates based
    on a power function.

    Transformation
    --------------
    r' = r^{1 + strength}
    x' = r' cos(theta)
    y' = r' sin(theta)
    with r and theta the polar form coordinate of x,y and k the strenght.

    Parameters
    ----------
    strength : float
        The strength of the distorsion. Identity function for strenght = 0.0.
    """
    def __init__(self, strength=0.5):
        super(Fisheye, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the fisheye distortion.

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
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        # get polar form
        magnitude = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        angle = torch.atan2(y_indices, x_indices)
        # stretch magnitude
        magnitude_new = magnitude ** (1 + self.strength)
        x_new = magnitude_new * torch.cos(angle)
        y_new = magnitude_new * torch.sin(angle)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Swirl(WarpTransform):
    """
    Applies a swirl distortion effect by rotating the coordinates around the center
    based on their radial distance from the center.

    Transformation
    --------------
    theta' = theta + (radius - r) * strength
    x' = r' * cos(theta')
    y' = r' * sin(theta')

    Parameters
    ----------
    strength : float
        The strength of the swirl effect. Higher values result in a stronger swirl.
    radius : float
        The maximum distance from the center where the swirl effect is applied.
    """
    def __init__(self, strength=1.0, radius=1.0):
        super(Swirl, self).__init__()
        self.strength = strength
        self.radius = radius

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the swirl distortion.

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
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        # get polar form
        magnitude = torch.sqrt(x_indices**2 + y_indices**2)
        angle = torch.atan2(y_indices, x_indices)
        # rotate anticlockwise depending on the radius and strength
        swirl_factor = (self.radius - magnitude) * self.strength
        angle += swirl_factor

        x_new = magnitude * torch.cos(angle)
        y_new = magnitude * torch.sin(angle)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class BarrelDistortion(WarpTransform):
    """
    Applies a barrel distortion effect by distorting the radial distance from
    the center using a polynomial approximation.

    Note: We use an an approximation of the polynomial radial distortion model to invert the
    original barrel formulas.
    see https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

    Transformation
    --------------
    r' = r * (1 + strength * r^2)
    x' = r' * cos(theta)
    y' = r' * sin(theta)

    Parameters
    ----------
    strength : float
        The strength of the barrel distortion effect.
    """
    def __init__(self, strength=0.5):
        super(BarrelDistortion, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the barrel distortion.

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
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        # compute the undistorted radial distance r_u
        r_u = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        # compute the distorted radial distance r_d
        r_d = r_u * (1 + self.strength * r_u ** 2)
        # angle dont' change
        x_new = r_d * torch.cos(theta)
        y_new = r_d * torch.sin(theta)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Pinchcursion(BarrelDistortion):
    """
    Applies a pincushion distortion effect, which is the inverse of barrel distortion.

    Transformation
    --------------
    r' = r * (1 - strength * r^2)
    x' = r' * cos(theta)
    y' = r' * sin(theta)

    Parameters
    ----------
    strength : float
        The strength of the pincushion distortion effect.
    """
    def __init__(self, strength=0.5):
        super(Pinchcursion, self).__init__(-strength)

class Stretching(WarpTransform):
    """
    Applies a stretching effect along the specified axis.

    Transformation
    --------------
    x' = x * (1 + strength)  (if axis = 'x')
    y' = y * (1 + strength)  (if axis = 'y')

    Parameters
    ----------
    strength : float
        The strength of the stretching effect.
    axis : str
        The axis along which to apply the stretching effect ('horizontal' or 'vertical').
    """
    def __init__(self, strength=0.5, axis='horizontal'):
        super(Stretching, self).__init__()
        assert axis in ['x', 'y']

        self.strength = strength
        self.axis = axis

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        if self.axis == 'x':
            x_new = x_indices * (1 + self.strength)
            y_new = y_indices
        else:
            x_new = x_indices
            y_new = y_indices * (1 + self.strength)

        # create the grid for warping
        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Twirl(WarpTransform):
    """
    Applies a twirl distortion effect by rotating the coordinates around the center
    with an angle proportional to their radial distance from the center.

    Transformation
    --------------
    theta' = theta + strength * r
    x' = r * cos(angle')
    y' = r * sin(angle')

    Parameters
    ----------
    strength : float
        The strength of the twirl effect.
    """
    def __init__(self, strength=0.5):
        super(Twirl, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        magnitude = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        angle = torch.atan2(y_indices, x_indices)

        # twirl increase angle shift depending monotonically linearly on the distance
        twirl_shift = self.strength * magnitude

        x_new = magnitude * torch.cos(angle + twirl_shift)
        y_new = magnitude * torch.sin(angle + twirl_shift)

        # create the grid for warping
        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Wave(WarpTransform):
    """
    Applies a wave distortion effect along the specified axis.

    Transformation
    --------------
    y' = y + amplitude * sin(2 * pi * frequency * x + phase)  (if axis = 'x')
    x' = x + amplitude * sin(2 * pi * frequency * y + phase)  (if axis = 'y')

    Parameters
    ----------
    amplitude : float
        The amplitude of the wave.
    frequency : float
        The frequency of the wave.
    phase : float
        The phase shift of the wave.
    axis : str
        The axis along which to apply the wave effect ('x' or 'y').
    """
    def __init__(self, amplitude=0.1, frequency=1.0, phase=0.0, axis='x'):
        assert axis in ['x', 'y']

        super(Wave, self).__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.axis = axis

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        if self.axis == 'x':
            x_new = x_indices
            y_new = y_indices + self.amplitude * torch.sin(2 * torch.pi * self.frequency * x_indices + self.phase)
        else:
            x_new = x_indices + self.amplitude * torch.sin(2 * torch.pi * self.frequency * y_indices + self.phase)
            y_new = y_indices

        # create the grid for warping
        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Spherize(WarpTransform):
    """
    Applies a spherical distortion effect by mapping the coordinates onto a sphere.

    Transformation
    --------------
    r' = sin(r * strength * pi/2) / (strength * pi/2)
    x' = r' * cos(theta)
    y' = r' * sin(theta)

    Parameters
    ----------
    strength : float
        The strength of the spherical distortion effect.
    """
    def __init__(self, strength=0.5):
        super(Spherize, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        magnitude = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        angle = torch.atan2(y_indices, x_indices)

        magnitude_new = torch.sin(magnitude * self.strength * torch.pi / 2) / (self.strength * torch.pi / 2)

        x_new = magnitude_new * torch.cos(angle)
        y_new = magnitude_new * torch.sin(angle)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Bulge(WarpTransform):
    """
    Applies a bulge distortion effect by radially distorting the coordinates.

    Transformation
    --------------
    r' = r^strength
    x' = r' * cos(theta)
    y' = r' * sin(theta)

    Parameters
    ----------
    strength : float
        The strength of the bulge distortion effect. Must be greater than or equal to 1.0.
    """
    def __init__(self, strength=1.0):
        super(Bulge, self).__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        y_indices, x_indices = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

        # compute the radial distance from the center
        r = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        # apply bulge distortion
        r_new = r ** self.strength

        # compute new coordinates
        x_new = r_new * torch.cos(theta)
        y_new = r_new * torch.sin(theta)

        # create the grid for warping
        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid

class Implosion(Bulge):
    """
    Applies an implosion distortion effect by radially distorting the coordinates inward.

    Transformation
    --------------
    r' = r^strength
    x' = r' * cos(theta)
    y' = r' * sin(theta)

    Parameters
    ----------
    strength : float
        The strength of the implosion distortion effect. Must be between 0.0 and 1.0.
    """
    def __init__(self, strength=0.5):
        assert 0.0 < strength <= 1.0
        super(Implosion, self).__init__(strength)

