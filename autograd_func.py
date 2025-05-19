from typing import Tuple, Any

import torch


def compute_grad_torch():
    torch.manual_seed(42)
    x = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)
    y = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)

    # autorag op
    f = torch.exp(x) + torch.cos(y)
    # autograd op end

    loss = f.sum(dim=0).var(dim=0)
    loss.backward()
    return x.grad, y.grad


class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx:Any, x: torch.Tensor, y: torch.Tensor):
        exp_x = torch.exp(x)
        ctx.save_for_backward(exp_x, y)

        return exp_x + torch.cos(y)

    @staticmethod
    def backward(ctx: Any, grad_output_f: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        exp_x, y = ctx.saved_tensors

        dx = exp_x
        dy = -torch.sin(y)

        return grad_output_f * dx, grad_output_f * dy


def my_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return MyFunc.apply(x, y)


def compute_grad_custom():
    torch.manual_seed(42)
    x = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)
    y = torch.randn((15, 10), dtype=torch.float32, requires_grad=True)

    # autorag op
    f = my_func(x, y)
    # autograd op end

    loss = f.sum(dim=0).var(dim=0)
    loss.backward()
    return x.grad, y.grad

if __name__ == '__main__':
    gradx_torch, grady_torch = compute_grad_torch()
    gradx_custom, grady_custom = compute_grad_custom()
    assert torch.allclose(gradx_custom, gradx_torch, atol=0.00001)
    assert torch.allclose(grady_custom, grady_torch, atol=0.00001)