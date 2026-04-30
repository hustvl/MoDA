import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable


class BiClipSGD_Full(Optimizer):
    def __init__(self, params, lr=1e-3, nu=0.0, d=1.0, gamma=0.0, u=float('inf'),
                 zeta=0.0, l2_or_coord_clip=0, weight_decay=0.0, global_epoch=1):
        """
        Initializes the BiClipSGD_Full optimizer.

        Adapted from https://github.com/sulee3/Heavy_Tails/blob/main/models/ClipUpdate.py

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr (float): base learning rate (client_lr in original implementation)
            nu (float): exponent for learning rate scaling
            d (float): lower clipping threshold base value
            gamma (float): exponent for lower clipping threshold
            u (float): upper clipping threshold base value
            zeta (float): exponent for upper clipping threshold
            l2_or_coord_clip (int): 0 for L2 clipping, 1 for coordinate-wise clipping
            weight_decay (float): weight decay coefficient
            global_epoch (int): current epoch (>=1)
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if l2_or_coord_clip not in [0, 1]:
            raise ValueError("l2_or_coord_clip must be 0 (L2) or 1 (coordinate-wise)")

        defaults = dict(lr=lr, nu=nu, d=d, gamma=gamma, u=u, zeta=zeta,
                       l2_or_coord_clip=l2_or_coord_clip, weight_decay=weight_decay,
                       global_epoch=global_epoch)
        super(BiClipSGD_Full, self).__init__(params, defaults)

        # Optional logging/tracking attributes
        self.last_grad_norm = None
        self.last_scaling_factor = None

    def step(self, closure: Optional[Callable[[], float]] = None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Ensure t (global epoch) is at least 1
            t = max(group['global_epoch'], 1)
            lr = group['lr'] * (t ** group['nu'])
            lower_clip = group['d'] * (t ** group['gamma'])
            upper_clip = group['u'] * (t ** group['zeta'])
            clip_mode = group['l2_or_coord_clip']
            weight_decay = group['weight_decay']

            params_with_grad = []
            grads = []

            # Collect parameters with gradients
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

            if not params_with_grad:
                continue

            with torch.no_grad():
                if clip_mode == 0:
                    # L2 Clipping: compute global gradient norm
                    total_norm_sq = torch.tensor(0.0, device=grads[0].device, dtype=grads[0].dtype)
                    for grad in grads:
                        total_norm_sq += grad.data.pow(2).sum()
                    grad_norm = total_norm_sq.sqrt().item()
                    self.last_grad_norm = grad_norm

                    if grad_norm == 0:
                        continue  # No update if all gradients are zero

                    if grad_norm < lower_clip:
                        scaling_factor = lower_clip / grad_norm
                    elif grad_norm > upper_clip:
                        scaling_factor = upper_clip / grad_norm
                    else:
                        scaling_factor = 1.0
                    self.last_scaling_factor = scaling_factor

                    # Scale each gradient and update parameters
                    for param, grad in zip(params_with_grad, grads):
                        grad.data.mul_(scaling_factor)
                        param.data.add_(grad.data, alpha=-lr)
                        if weight_decay > 0:
                            param.data.mul_(1 - lr * weight_decay)

                elif clip_mode == 1:
                    # Coordinate-wise Clipping
                    for param, grad in zip(params_with_grad, grads):
                        # Save sign before modifying grad in-place
                        sign = grad.data.sign()
                        grad.data.abs_()  # In-place absolute value
                        grad.data.clamp_(min=lower_clip, max=upper_clip)  # In-place clamp
                        grad.data.mul_(sign)  # Restore original sign in-place
                        param.data.add_(grad.data, alpha=-lr)
                        if weight_decay > 0:
                            param.data.mul_(1 - lr * weight_decay)
                else:
                    raise ValueError(f"Unsupported clipping mode: {clip_mode}. Use 0 for L2 or 1 for Coordinate-wise.")

        return loss

    def update_global_epoch(self, new_epoch):
        """
        Updates the global epoch for all parameter groups.

        Args:
            new_epoch (int): The new global epoch value.
        """
        for group in self.param_groups:
            group['global_epoch'] = new_epoch
