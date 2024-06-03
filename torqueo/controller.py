import torch
import torch.nn.functional as F

class WarpController(): 
    """
    This class wraps the WarpTransform subclass by making it an object of this class. 
    The user can now input distortion levels from 0 to 5 and they will automatically be translated to corresponding strength of distortion arguments for warps. 

    Parameters
    ----------
    warp_transform: WarpTransform
        The particular warping class that we want a controller over. Example: Fisheye, Swirl. 

    Example Usage
    -------------
    SwirlController = WarpController(Swirl)
    swirl_transformer = SwirlController(1, radius=1) #or, SwirlController(distortion_level=1, radius=1)

    The args following distortion_level need to be keyword arguments. 
    """
    def __init__(self, warp_transform):
        self.warp_transform = warp_transform
        self.class_name = warp_transform.__name__
        self.mapping_dict = {"Fisheye": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Swirl": lambda x: torch.linspace(0, 1, 6)[x], 
                        "BarrelDistortion": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Pincushion": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Stretching": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Compression": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Twirl": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Wave": lambda x: torch.linspace(1, 4, 6)[x], 
                        "Spherize": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Bulge": lambda x: torch.linspace(1, 3, 6)[x], 
                        "Implosion": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Pinch": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Punch": lambda x: torch.linspace(0, 1, 6)[x], 
                        "Shear": lambda x: torch.linspace(0, 1, 6)[x], 
                        "PerspectiveWarp": lambda x: torch.linspace(0, 1, 6)[x] 
                        }

    def __call__(self, distortion_level=0, **kwargs): 
        """
        Parameters
        ----------
        distortion_level: int
            The distortion effect from 0 to 5. 0 for identify function, 5 for max distortion as defined in the mapping function. 

        Returns
        -------
        WarpTransform
            The initialized object of WarpTransform with the given distortion level and keyword arguments. 
        
        Example Usage
        -------------
        swirl_transformer = SwirlController(1, radius=1) #or, SwirlController(distortion_level=1, radius=1) 
        Args following distortion_level need to be keyword arguments. 
        """
        strength = self.mapping_dict[self.class_name](distortion_level)
        return self.warp_transform(strength=strength, **kwargs)