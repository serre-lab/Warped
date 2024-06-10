import torch


class WarpController():
    """
    This class wraps the WarpTransform subclasses by making them an object of this class.
    The distortion levels are mapped to a given range of strengths for each warp
    to allow a common interface across all warping transformations.

    Parameters
    ----------
    warp_transform: WarpTransform
        The particular warping class that we want a controller over. Example: Fisheye, Swirl.
    max_distortion_level: int
        The maximum distortion level possible for this warping controller. Should be >= 0.
        When max_distortion_level is not passed with a range, the strength range is hard-coded
        and the max_distortion_level is used to determine how many samples to get from this range.
    strength_range: A tuple of floats or a tuple of ints
        The minimum/identity warping strength and the maximum warping strength for a transformation.
        We uniformly sample max_distortion_level+1 levels from this range.

    Example Usage
    -------------
    SwirlController = WarpController(Swirl, 5, (0, 1))
    swirl_transformer = SwirlController(1, radius=1)

    The args following distortion_level need to be keyword arguments.
    """

    def __init__(self, warp_transform, max_distortion_level=10, strength_range=None):
        assert max_distortion_level >= 0
        assert strength_range is None or len(strength_range) == 2
        self.warp_transform = warp_transform
        self.class_name = warp_transform.__name__
        self.max_distortion_level = max_distortion_level
        total_levels = max_distortion_level + 1
        self.mapping_dict = {
            "Fisheye": lambda x: torch.linspace(0, 1, total_levels)[x],
            "Swirl": lambda x: torch.linspace(0, 1, total_levels)[x],
            "BarrelDistortion": lambda x: torch.linspace(0, 1, total_levels)[x],
            "Pincushion": lambda x: torch.linspace(0, 0.9, total_levels)[x],
            "Stretching": lambda x: torch.linspace(0, 0.8, total_levels)[x],
            "Compression": lambda x: torch.linspace(0, 2, total_levels)[x],
            "Twirl": lambda x: torch.linspace(0, 1, total_levels)[x],
            "Wave": lambda x: torch.linspace(0, 1.8, total_levels)[x],
            "Spherize": lambda x: torch.linspace(0, 1.2, total_levels)[x],
            "Bulge": lambda x: torch.linspace(1, 3, total_levels)[x],
            "Implosion": lambda x: torch.linspace(0.5, 1, total_levels)[total_levels -1 -x],
            "Pinch": lambda x: torch.linspace(0, 0.6, total_levels)[x],
            "Punch": lambda x: torch.linspace(0, 0.7, total_levels)[x],
            "Shear": lambda x: torch.linspace(0, 1.2, total_levels)[x],
            "Perspective": lambda x: torch.linspace(0, 0.9, total_levels)[x]
        }
        if strength_range:
            if self.class_name == "Implosion":
                assert strength_range[1] == 1, "Identity transformation for implosion."
                self.mapping_dict[self.class_name] = lambda x: (
                    torch.linspace(
                        strength_range[0], strength_range[1], total_levels
                    )[total_levels -1 -x]
                )
            else:
                self.mapping_dict[self.class_name] = lambda x: (
                    torch.linspace(strength_range[0], strength_range[1], total_levels)[x]
                )

    def __call__(self, distortion_level=0, **kwargs):
        """
        Parameters
        ----------
        distortion_level: int
            The distortion effect from 0 to max_distortion_level. 0 for identify function,
            max_distortion_level for max distortion as defined in the mapping function.

        Returns
        -------
        WarpTransform
            The initialized object of WarpTransform with the given distortion level
            and keyword arguments.
        """
        assert 0 <= distortion_level <= self.max_distortion_level
        strength = self.mapping_dict[self.class_name](distortion_level)
        return self.warp_transform(strength=strength, **kwargs)
