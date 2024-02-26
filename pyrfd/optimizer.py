import torch
from torch.optim.optimizer import Optimizer

from .covariance import IsotropicCovariance


class RFD(Optimizer):
    """Random Function Descent (RFD) optimizer"""

    def __init__(self, params, *, covariance_model: IsotropicCovariance, momentum=0):
        defaults = dict(cov=covariance_model, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure):
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

                cov_model: IsotropicCovariance = group["cov"]
                lr = cov_model.learning_rate(loss, grad_norm)

                for param in group["params"]:
                    state = self.state[param]
                    if param.grad is not None:
                        if "velocity" in state.keys():
                            velocity = state["velocity"]
                            velocity.mul_(momentum).add_(param.grad, alpha=-1)
                            # multiply the current velocity by momentum parameter and subtract the current gradient (in-place)
                        else:
                            velocity = torch.mul(param.grad, -1)

                        param += lr * velocity  # add velocity to parameters in-place!
                        state["velocity"] = velocity

        return loss
