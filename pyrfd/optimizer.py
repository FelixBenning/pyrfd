""" Random Function Descent (RFD) optimizer implementing the pytorch optimizer
interface
"""

import torch
from torch.optim.optimizer import Optimizer

from .covariance import IsotropicCovariance


class RFD(Optimizer):
    """Random Function Descent (RFD) optimizer

    To enable the usage of step size schedulers, the `lr` parameter is multiplied
    to the statistically determined step size (i.e. default `1`)
    """

    def __init__(
        self,
        params,
        *,
        covariance_model: IsotropicCovariance,
        momentum=0,
        lr=1,
        b_size_inv=0,
        norm_lock=False,
    ):
        defaults = {
            "cov": covariance_model,
            "momentum": momentum,
            "lr": lr,  # really a learning rate multiplier,
            # but this name ensures compatibility with schedulers
            "learning_rate": None,
            "b_size_inv": b_size_inv,
            "norm_lock": norm_lock,
        }

        super().__init__(params, defaults)

    def step(self, closure):  # pylint: disable=locally-disabled, signature-differs
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        with torch.enable_grad():
            loss = closure()
            # we assume here, that the person using the optimizer has used
            # optimizer.zero_grad before. Calling the closure with torch.enable_grad()
            # autograd to build a tree which we can

        with torch.no_grad():
            for group in self.param_groups:
                grads = [
                    param.grad.detach().flatten()
                    for param in group["params"]
                    if param.grad is not None
                ]
                grad_norm = torch.cat(grads).norm()

                momentum = group["momentum"]
                lr_multiplier = group["lr"]
                b_size_inv = group["b_size_inv"]
                norm_lock = group["norm_lock"]

                cov_model: IsotropicCovariance = group["cov"]
                learning_rate = lr_multiplier * cov_model.learning_rate(
                    loss, grad_norm, b_size_inv=b_size_inv
                )
                group["learning_rate"] = learning_rate

                param: torch.Tensor
                for param in group["params"]:
                    state = self.state[param]
                    if param.grad is not None:
                        if "velocity" in state.keys():
                            velocity = state["velocity"]
                            velocity.mul_(momentum).add_(param.grad, alpha=-1)
                            # multiply the current velocity by momentum
                            # parameter and subtract the current gradient (in-place)
                        else:
                            velocity = torch.mul(param.grad, -1)

                        if norm_lock:
                            param_norm = param.norm()

                        # add velocity to parameters in-place!
                        param += learning_rate * velocity

                        if norm_lock:
                            # project back to the original sphere
                            param.mul_(param_norm / param.norm())

                        state["velocity"] = velocity

        return loss
