import torch

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
        super().__init__()
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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

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
        super().__init__()
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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

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
        super().__init__()
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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        # compute the undistorted radial distance r_u
        r_u = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        # compute the distorted radial distance r_d
        r_d = r_u * (1 + self.strength * r_u ** 2)
        # angle dont' change
        x_new = r_d * torch.cos(theta)
        y_new = r_d * torch.sin(theta)
        # scale to fill image
        x_scale = min(1, 1/torch.max(torch.abs(x_new)))
        y_scale = min(1, 1/torch.max(torch.abs(y_new)))
        x_new = x_new*x_scale
        y_new = y_new*y_scale

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid


class Pincushion(BarrelDistortion):
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
        super().__init__(-strength)


class Stretching(WarpTransform):
    """
    Applies a stretching effect along the specified axis.

    Transformation
    --------------
    x' = x * (1 - strength)  (if axis = 'horizontal')
    y' = y * (1 - strength)  (if axis = 'vertical')

    Parameters
    ----------
    strength : float
        The strength of the stretching effect.
    axis : str
        The axis along which to apply the stretching effect ('horizontal' or 'vertical').
    """

    def __init__(self, strength=0.5, axis='horizontal'):
        super().__init__()
        assert axis in ['horizontal', 'vertical']

        self.strength = strength
        self.axis = axis

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the stretching distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        if self.axis == 'horizontal':
            x_new = x_indices * (1 - self.strength)
            y_new = y_indices
        else:
            x_new = x_indices
            y_new = y_indices * (1 - self.strength)

        # create the grid for warping
        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid


class Compression(Stretching):
    """
    Applies a compression effect along the specified axis.

    Transformation
    --------------
    x' = x * (1 + strength)  (if axis = 'horizontal')
    y' = y * (1 + strength)  (if axis = 'vertical')

    Parameters
    ----------
    strength : float
        The strength of the stretching effect. Should be >=0.
    axis : str
        The axis along which to apply the stretching effect ('horizontal' or 'vertical').
    """

    def __init__(self, strength=0.5, axis='horizontal'):
        assert strength >= 0
        super().__init__(-strength, axis)


class Twirl(WarpTransform):
    """
    Applies a twirl distortion effect by rotating the coordinates around the center
    with an angle proportional to their radial distance from the center.

    Transformation
    --------------
    theta' = theta + strength * r
    x' = r * cos(theta')
    y' = r * sin(theta')

    Parameters
    ----------
    strength : float
        The strength of the twirl effect.
    """

    def __init__(self, strength=0.5):
        super().__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the twirl distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

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
    y' = y + strength * amplitude * sin(2 * pi * frequency * x + phase)  (if axis = 'horizontal')
    x' = x + strength * amplitude * sin(2 * pi * frequency * y + phase)  (if axis = 'vertical')

    Parameters
    ----------
    strength: float
        The strength of the distortion as defined by the scaling of the amplitude.
    amplitude : float
        The amplitude of the wave.
    frequency : float
        The frequency of the wave.
    phase : float
        The phase shift of the wave.
    axis : str
        The axis along which to apply the wave effect ('horizontal' or 'vertical').
    """

    def __init__(self, strength=1, amplitude=0.1, frequency=1.0, phase=0.0, axis='horizontal'):
        super().__init__()
        self.strength = strength
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.axis = axis

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the wave distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        if self.axis == 'horizontal':
            x_new = x_indices
            y_new = y_indices + self.strength * self.amplitude * torch.sin(
                2 * torch.pi * self.frequency * x_indices + self.phase)
        else:
            x_new = x_indices + self.strength * self.amplitude * torch.sin(
                2 * torch.pi * self.frequency * y_indices + self.phase)
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
        super().__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the spherize distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        magnitude = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        angle = torch.atan2(y_indices, x_indices)

        magnitude_new = torch.sin(magnitude * self.strength * torch.pi / 2) / (
            self.strength * torch.pi / 2) if self.strength else magnitude

        x_new = magnitude_new * torch.cos(angle)
        y_new = magnitude_new * torch.sin(angle)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)

        return grid


class Bulge(WarpTransform):
    """
    Applies a bulge distortion effect by radially distorting the coordinates outward.

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
        super().__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the bulge distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        # compute the radial distance from the center
        magnitude = torch.sqrt(x_indices ** 2 + y_indices ** 2)
        theta = torch.atan2(y_indices, x_indices)

        # apply bulge distortion
        magnitude_new = magnitude ** self.strength

        # compute new coordinates
        x_new = magnitude_new * torch.cos(theta)
        y_new = magnitude_new * torch.sin(theta)

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
        super().__init__(strength)


class Pinch(WarpTransform):
    """
    Applies a pinch distortion effect by pinching it towards the user.

    Transformation
    --------------
    pinch_factor = sin(pi/2 * r)^(-strength)
    x' = x' * pinch_factor
    y' = y' * pinch_factor

    Parameters
    ----------
    strength : float
        The strength of the pinch distortion effect. Must be between 0.0 and 1.0.
    """

    def __init__(self, strength=0.5):
        super().__init__()
        assert 0.0 <= abs(strength) <= 1.0
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the pinch distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        magnitude = torch.sqrt(x_indices ** 2 + y_indices ** 2)

        pinch_factor = torch.sin((torch.pi/2) * magnitude) ** (-1*self.strength)
        x_new = x_indices*pinch_factor
        y_new = y_indices*pinch_factor

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid


class Punch(Pinch):
    """
    Applies a punch distortion effect which is opposite of pinch and pushes
    image away from the user.

    Transformation
    --------------
    punch_factor = sin(pi/2 * r)^(strength)
    x' = x' * punch_factor
    y' = y' * punch_factor

    Parameters
    ----------
    strength : float
        The strength of the punch distortion effect. Must be between 0.0 and 1.0.
    """

    def __init__(self, strength=0.5):
        super().__init__(-strength)


class Shear(WarpTransform):
    """
    Applies a shear distortion effect.
    Implemented as defined here: https://en.wikipedia.org/wiki/Shear_mapping

    Transformation
    --------------
    x' = x + strength*y (if axis = 'horizontal')
    y' = strength*x + y (if axis = 'vertical')

    Parameters
    ----------
    strength : float
        The strength of the shear distortion effect. Also defined as the cotangent of
        the shear angle. Positive for right/top leaning, negative for left/bottom leaning shear.
    axis: str
        The axis along which to apply the shearing effect ('horizontal' or 'vertical').
    """

    def __init__(self, strength=0.5, axis='horizontal'):
        super().__init__()
        self.strength = strength
        assert axis in ['horizontal', 'vertical']
        self.axis = axis

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the shear distortion.

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
        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        if self.axis == 'horizontal':
            x_new = x_indices + self.strength*y_indices
            y_new = y_indices
        else:
            x_new = x_indices
            y_new = self.strength*x_indices + y_indices

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid


class Perspective(WarpTransform):
    """
    # pylint: disable=C0301

    Applies a random perspective wrap as defined here:
    https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomPerspective
    https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#perspective

    Transformation
    --------------
    Picks random end points for perspective warp and uses current corners as start points
    to generate the warping matrix. Given the warping coefficients (a, b, c, d, e, f, g, h)
    we have the following association:
    x' = (ax + by + c) / (gx + hy + 1)
    y' = (dx + ey + f) / (gx + hy + 1)

    Parameters
    ----------
    strength : float
        The strength of the perspective distortion effect.
    """

    def __init__(self, strength=0.5):
        super().__init__()
        self.strength = strength

    def generate_warp_field(self, height, width):
        """
        Generates the warp field for the shear distortion.

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
        # pylint: disable=C0103

        y_indices, x_indices = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        distortion_scale = self.strength

        topleft = [
            -1 + distortion_scale*torch.rand(1).item(),
            -1 + distortion_scale*torch.rand(1).item(),
        ]
        topright = [
            1 - distortion_scale*torch.rand(1).item(),
            -1 + distortion_scale*torch.rand(1).item(),
        ]
        botright = [
            1 - distortion_scale*torch.rand(1).item(),
            1 - distortion_scale*torch.rand(1).item(),
        ]
        botleft = [
            -1 + distortion_scale*torch.rand(1).item(),
            1 - distortion_scale*torch.rand(1).item(),
        ]
        startpoints = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        endpoints = [topleft, topright, botright, botleft]

        a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float64)
        for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
            a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0,
                                              0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0,
                                                  p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        b_matrix = torch.tensor(startpoints, dtype=torch.float64).view(8)
        res = torch.linalg.lstsq(
            a_matrix, b_matrix, driver="gels").solution.to(
            torch.float32).tolist()
        # res is (a, b, c, d, e, f, g, h)
        # (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )
        x_new = (res[0]*x_indices + res[1]*y_indices + res[2]
                 )/(res[6]*x_indices + res[7]*y_indices + 1)
        y_new = (res[3]*x_indices + res[4]*y_indices + res[5]
                 )/(res[6]*x_indices + res[7]*y_indices + 1)

        grid = torch.stack((x_new, y_new), dim=-1)
        grid = grid.unsqueeze(0)
        return grid
