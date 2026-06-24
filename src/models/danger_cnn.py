"""
DangerCNN: V(x, g) = P(goal | x, g) from a rasterized pitch.

Channels (in order): shooter, ball, attacking_teammates, defenders_excl_gk,
goalkeeper, then (optionally) 5 static goal-geometry channels. Input
(B, in_channels, 80, 60), output (B,) probability in [0, 1].

`pool_size` controls the adaptive-average-pool grid before the head. It may be an
int (square) or an (h, w) tuple. pool_size=1 (default, baseline_v1) collapses the
whole pitch to a per-channel average, which discards absolute location; a larger
grid (baseline_v2 uses (4, 3)) keeps a coarse spatial layout so the head can
reason about *where* signals are, not just how much. Note the feature map before
this pool is 20x15, and MPS requires the output to divide it evenly — (4, 3) does
(20/4, 15/3); a square 3 or 4 does not.

Use `forward_logits(x)` for training (paired with BCEWithLogitsLoss for
numerical stability) and `forward(x)` for inference (returns probabilities).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DangerCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        dropout_p: float = 0.3,
        pool_size: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        # Normalize to an (h, w) tuple; JSON round-trips pool_size as a list.
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pool_size: tuple[int, int] = (int(pool_size[0]), int(pool_size[1]))

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * self.pool_size[0] * self.pool_size[1], 64)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # He (Kaiming) for Conv2d to match downstream ReLU; Xavier for Linear.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.adaptive_avg_pool2d(x, self.pool_size).flatten(1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


def _print_param_breakdown(model: nn.Module) -> int:
    print(f"{'Parameter':<28}{'Shape':<22}{'Count':>12}")
    print("-" * 62)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"{name:<28}{str(tuple(p.shape)):<22}{n:>12,}")
    print("-" * 62)
    print(f"{'TOTAL':<28}{'':<22}{total:>12,}")
    return total


def _approx_flops_per_shot() -> int:
    """Rough FLOPs (mul+add as 2) for one forward pass at batch size 1.

    Pooling, BN, and elementwise ops are ignored — the conv layers dominate.
    """
    def conv_flops(c_in, c_out, h_out, w_out, k):
        return 2 * c_in * c_out * h_out * w_out * k * k

    flops = 0
    flops += conv_flops(5, 32, 80, 60, 3)
    flops += conv_flops(32, 64, 40, 30, 3)
    flops += conv_flops(64, 128, 20, 15, 3)
    flops += 2 * 128 * 64
    flops += 2 * 64 * 1
    return flops


if __name__ == "__main__":
    torch.manual_seed(0)

    model = DangerCNN()
    total = _print_param_breakdown(model)
    in_range = 100_000 <= total <= 200_000
    print(
        f"\nTotal: {total:,} params  "
        f"({'within' if in_range else 'OUTSIDE'} target 100k-200k range)"
    )

    # Forward pass on a random batch.
    x = torch.randn(4, 5, 80, 60)
    model.eval()
    with torch.no_grad():
        probs = model(x)
        logits = model.forward_logits(x)
    assert probs.shape == (4,), f"forward shape {tuple(probs.shape)}"
    assert logits.shape == (4,), f"forward_logits shape {tuple(logits.shape)}"
    assert (probs >= 0).all() and (probs <= 1).all(), "probabilities out of [0,1]"
    print(
        f"\nforward(x):        shape={tuple(probs.shape)} "
        f"range=[{probs.min():.3f}, {probs.max():.3f}]"
    )
    print(
        f"forward_logits(x): shape={tuple(logits.shape)} "
        f"range=[{logits.min():.3f}, {logits.max():.3f}]"
    )

    # Forward + backward with BCEWithLogitsLoss to verify all params receive grads.
    model.train()
    logits = model.forward_logits(x)
    target = torch.randint(0, 2, (4,), dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    missing = []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(f"{name} (no grad)")
        elif not torch.isfinite(p.grad).all():
            missing.append(f"{name} (non-finite grad)")
        elif p.grad.abs().sum().item() == 0:
            missing.append(f"{name} (zero grad)")
    n_param_tensors = sum(1 for _ in model.parameters())
    if missing:
        print(f"\nGradient check FAILED: {missing}")
    else:
        print(
            f"\nGradient check OK: all {n_param_tensors} parameter tensors "
            f"have finite, non-zero gradients (loss={loss.item():.4f})."
        )

    flops = _approx_flops_per_shot()
    print(
        f"\nApprox FLOPs per shot (forward, batch=1): "
        f"{flops:,} (~{flops / 1e6:.1f} MFLOPs)"
    )
