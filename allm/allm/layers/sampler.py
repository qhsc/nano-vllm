import torch
from torch import nn


class Sampler(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(self.dtype)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=self.dtype)
        epsilon = 1e-10
        gumbel_noise = torch.empty_like(probs).exponential_(1) + epsilon
        sampled_tokens = probs.div_(gumbel_noise).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sampled_tokens)
