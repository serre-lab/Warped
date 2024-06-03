"""
Torqueo: Personal toolbox for experimenting with image warping
-------

This little personnal toolbox was created to help us play with image
warping transformations.

"""

__version__ = '0.0.0'

from .base import WarpTransform
from .plots import show
from .transforms import (BarrelDistortion, Bulge, Fisheye, Perspective, Pinch, Spherize,
                         Stretching, Swirl, Twirl, Wave, Punch, Implosion, Compression)