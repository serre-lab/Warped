"""
Torqueo: Personal toolbox for experimenting with image warping
-------

This little personnal toolbox was created to help us play with image
warping transformations.

"""

__version__ = '0.0.2'

from .base import WarpTransform
from .plots import show
from .transforms import (
    Fisheye, Swirl, BarrelDistortion, Pincushion, Stretching, Compression, Twirl, Wave, Spherize,
    Bulge, Implosion, Pinch, Punch, Shear, Perspective)
from .controller import WarpController
