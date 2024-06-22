import torch
from lomem import normalization



if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.float32
    shape = (31, 123, 767)
    eps = 1e-8

    x = (torch.rand(shape, dtype=dtype, device=device) - 0.5) * 2
    y_torch = torch.nn.functional.layer_norm(x, shape[-1:], eps=eps)
    y_lomem = normalization.layer_norm(x, shape[-1:], eps=eps)

    print((y_torch - torch.nn.functional.layer_norm(x.to(torch.float64), shape[-1:], eps=eps)).abs().mean())
    print((torch.nn.functional.layer_norm(x.to(torch.float64), shape[-1:], eps=eps) - y_lomem).abs().mean())

    print(y_torch[0, :10])
    print(y_lomem[0, :10])