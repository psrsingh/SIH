import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from train_depth import ssi_loss


def test_ssi_loss_recovers_scale_and_shift_with_grad():
    # y = 3*x + 2 exactly; the fitted residual should be ~0 regardless of x's
    # own scale, and the loss must stay differentiable w.r.t. the prediction.
    x_flat = torch.linspace(0, 1, 64, requires_grad=True)
    x = x_flat.reshape(1, 1, 8, 8)
    y = (3 * x.detach() + 2).reshape(1, 1, 8, 8)
    valid = torch.ones_like(y, dtype=torch.bool)

    loss = ssi_loss(x, y, valid)
    assert loss.item() < 1e-6

    loss.backward()
    assert x_flat.grad is not None


def test_ssi_loss_backward_does_not_allocate_quadratic_memory():
    # Regression test: torch.linalg.lstsq's backward pass over the full
    # valid-pixel vector previously allocated memory proportional to
    # pixel_count^2 (tens of GB at a 384x384 crop), which OOM'd real
    # training runs. A closed-form fit must stay linear in pixel count, so
    # this should run instantly even at a size that would have been
    # infeasible before the fix.
    n = 384
    pred = torch.rand(1, 1, n, n, requires_grad=True)
    target = torch.rand(1, 1, n, n)
    valid = torch.ones_like(target, dtype=torch.bool)

    loss = ssi_loss(pred, target, valid)
    loss.backward()
    assert pred.grad is not None
