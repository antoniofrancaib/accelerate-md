import torch
import numpy as np
from typing import Callable, Union
import copy

from .base import MCMCSampler

class LangevinDynamics(MCMCSampler):

    def __init__(self,
                 x: torch.Tensor,
                 energy_func: callable,
                 step_size: Union[float, torch.Tensor],
                 mh: bool = True,
                 device: str = 'cpu',
                 point_estimator: bool = False):
        """
        Standard Langevin Dynamics Sampler
        """
        super(LangevinDynamics, self).__init__(
            x=x,
            energy_func=energy_func,
            step_size=step_size,
            mh=mh,
            device=device,
            point_estimator=point_estimator
        )

        if self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]

            self.f_x = f_xc.detach()
            self.grad_x = grad_xc.detach()

    def sample(self) -> tuple:
        if self.point_estimator == True:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]
            x_p = x_c - self.step_size * grad_xc 
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), None

        if self.mh == False:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]

            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(x_c, device=self.device)
            
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), f_xc.detach()
        
        else:
            x_c = self.x.detach()
            f_xc = self.f_x.detach()
            grad_xc = self.grad_x.detach()

            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(self.x, device=self.device)
            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp = self.energy_func(x_p)
            grad_xp = torch.autograd.grad(f_xp.sum(), x_p,create_graph=False)[0]
            if isinstance(self.step_size, float):
                log_joint_prob_2 = -f_xc-torch.norm(x_p-x_c+self.step_size * grad_xc, dim=-1)**2/(4*self.step_size)
                log_joint_prob_1 = -f_xp-torch.norm(x_c-x_p+self.step_size * grad_xp, dim=-1)**2/(4*self.step_size)
            else:
                log_joint_prob_2 = -f_xc-torch.norm(x_p-x_c+self.step_size * grad_xc, dim=-1)**2/(4*self.step_size.squeeze(-1))
                log_joint_prob_1 = -f_xp-torch.norm(x_c-x_p+self.step_size * grad_xp, dim=-1)**2/(4*self.step_size.squeeze(-1))

            log_accept_rate = log_joint_prob_1 - log_joint_prob_2
            is_accept = torch.rand_like(log_accept_rate).log() <= log_accept_rate
            is_accept = is_accept.unsqueeze(-1)

            self.x = torch.where(is_accept, x_p.detach(), self.x)
            self.f_x = torch.where(is_accept.squeeze(-1), f_xp.detach(), self.f_x)
            self.grad_x = torch.where(is_accept, grad_xp.detach(), self.grad_x)

            acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp())

            return copy.deepcopy(self.x.detach()), acc_rate.detach()
        
