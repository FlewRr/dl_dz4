import torch
import torch.nn as nn
import torch.nn.functional as F

class MyRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        dims = tuple(-i for i in range(1, len(self.normalized_shape) + 1))
        rms = torch.rsqrt((x**2).mean(dim=dims, keepdim=True) + self.eps)

        x_norm = x * rms
        output = self.gamma * x_norm

        return output

if __name__ == "__main__":
    x = torch.randn(32, 128)
    myrmsnorm = MyRMSNorm(128)
    rmsnorm = nn.RMSNorm(128)

    my_y = myrmsnorm(x)
    y = rmsnorm(x)

    verdict = torch.allclose(y, my_y)

    # 3.0761704294945957e-10
    # True
    print((y-my_y).mean().item())
    print(verdict)