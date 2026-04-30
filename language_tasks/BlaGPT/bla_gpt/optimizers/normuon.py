import torch
import torch.distributed as dist

# copied from https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X



def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, ns_steps).float()
    ################ NorMuon added ###################
    vnorm = update.norm(dim=(-2,-1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
    ##################################################
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"], beta=group["momentum"], beta2=group["beta2"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class SingleDeviceNorMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p[...,0:1])
                update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"], beta=group["momentum"], beta2=group["beta2"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


# Hybrid optimizer combining NorMuon with AdamW fallback for non-2D parameters
class HybridNorMuon(torch.optim.Optimizer):
    """
    NorMuon optimizer with AdamW fallback for parameters that can't use NorMuon.

    NorMuon is used for 2D+ parameters (weight matrices), while AdamW is used for
    1D parameters (biases, norms) and embedding/output layers.

    Arguments:
        lr: Learning rate for NorMuon parameters (default: 0.02)
        weight_decay: Weight decay coefficient (default: 0.01)
        momentum: Momentum coefficient for NorMuon (default: 0.95)
        beta2: Second momentum coefficient for NorMuon (default: 0.95)
        nesterov: Use Nesterov momentum in NorMuon (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        normuon_params: Parameters to optimize with NorMuon
        adamw_params: Parameters to optimize with AdamW
        adamw_lr: Learning rate for AdamW (default: 3e-4)
        adamw_betas: Betas for AdamW (default: (0.9, 0.95))
        adamw_eps: Epsilon for AdamW (default: 1e-8)
    """

    def __init__(
        self,
        lr=0.02,
        weight_decay=0.01,
        normuon_params=None,
        momentum=0.95,
        beta2=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(normuon_params) if normuon_params else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Mark which parameters use NorMuon vs AdamW
        for p in normuon_params or []:
            assert p.ndim >= 2, f"NorMuon requires 2D+ parameters, got {p.ndim}D"
            self.state[p]["use_normuon"] = True
        for p in adamw_params:
            self.state[p]["use_normuon"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #         NorMuon          #
            ############################

            normuon_params = [p for p in group["params"] if self.state[p].get("use_normuon", False)]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in normuon_params:
                g = p.grad
                if g is None:
                    continue

                # Handle higher-dimensional tensors (e.g., conv filters)
                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # Initialize state
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["second_momentum_buffer"] = torch.zeros_like(g[..., 0:1])

                # Compute NorMuon update
                buf = state["momentum_buffer"]
                second_buf = state["second_momentum_buffer"]
                update = normuon_update(g, buf, second_buf, beta=momentum, beta2=beta2, ns_steps=ns_steps, nesterov=nesterov)

                # Apply weight decay and update
                p.mul_(1 - lr * weight_decay)
                p.add_(update.reshape(original_shape), alpha=-lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_params = [p for p in group["params"] if not self.state[p].get("use_normuon", False)]
            adamw_lr = group["adamw_lr"]
            beta1, beta2_adamw = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)

                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]

                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2_adamw)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2_adamw**step
                scale = bias_correction1 / bias_correction2**0.5

                p.mul_(1 - adamw_lr * weight_decay)
                p.add_(g, alpha=-adamw_lr / scale)

        return loss
