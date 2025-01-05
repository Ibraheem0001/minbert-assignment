from typing import Callable, Iterable, Tuple
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # todo

                # State should be stored in this dictionary
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                eps = group["eps"]
                _lambda = group['weight_decay']

                # Update first and second moments of the gradients
                beta_1, beta_2 = group['betas']
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2

                state["m"], state["v"] = m, v
                state["step"] += 1
                t = state["step"]

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if group['correct_bias']:
                    alpha_t = alpha * (math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t))
                else:
                    alpha_t = alpha

                # Update parameters
                # Please note: you should update p.data (not p), to avoid an error about a leaf Variable being used in an in-place operation
                p.data = p.data - alpha_t * (m / (v.sqrt() + eps)) #p.data represents model weights

                # Add weight decay after the main gradient-based updates.
                # Please note that, *unlike in https://arxiv.org/abs/1711.05101*, the learning rate should be incorporated into this update.
                # Please also note: you should update p.data (not p), to avoid an error about a leaf Variable being used in an in-place operation
                if group["weight_decay"] != 0:
                    p.data = p.data - (alpha * _lambda * p.data)

        return loss
