import torch
from torch.optim.optimizer import Optimizer
from typing import Optional

class RFDSqExp(Optimizer):
    def __init__(self, params, lr=0.03, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None: # need to obtain gradients ourselves
            with torch.enable_grad():
                loss = closure()
                # we assume here, that the person using the optimizer has used
                # optimizer.zero_grad before. Calling the closure with torch.enable_grad()
                # autograd to build a tree which we can

        with torch.no_grad():
            for group in self.param_groups:
                momentum = group["momentum"]
                lr = group["lr"]
                for param in group["params"]:
                    state = self.state[param]
                    if param.grad is not None:
                        if "velocity" in state.keys():
                            velocity = state["velocity"]
                            velocity.mul_(momentum).add_(param.grad, alpha=-1)
                            # multiply the current velocity by momentum parameter and subtract the current gradient (in-place)
                        else:
                            velocity = torch.clone(param.grad)
                        
                        param += lr * velocity # add velocity to parameters in-place!
                        state["velocity"] = velocity

        return loss