import torch
import numpy as np
from functools import partial
from typing import Optional, Callable
import copy
from torch.distributions import Normal, Gumbel
from typing import Union
import pdb


class MCMCSampler(object):
    def __init__(self,
                 x: torch.Tensor,
                 energy_func: Callable,
                 step_size: Union[float, torch.Tensor],
                 mh: bool = True,
                 device: str = 'cpu',
                 point_estimator: bool = False):
        
        self.x = x
        self.step_size = step_size
        self.energy_func = energy_func
        self.mh= mh
        self.device = device
        self.point_estimator = point_estimator

    def sample(self) -> tuple:
        pass
