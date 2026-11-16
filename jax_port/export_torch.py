"""Run with the PyTorch environment: export a PyTorch prior's weights and a reference output.

  python export_torch.py [--weights runs/prior/ema.pt] --out jax_port/ref.npz
Without --weights a randomly initialised network is exported (enough to test the port).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import UNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", type=Path)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
torch.manual_seed(0)
net = UNet().eval()
if a.weights:
    net.load_state_dict(torch.load(a.weights, map_location="cpu"))
x = torch.randn(2, 1, 160, 160)
t = torch.tensor([17, 640])
with torch.no_grad():
    y = net(x, t)
sd = {k: v.numpy() for k, v in net.state_dict().items()}
np.savez(a.out, **{"w/" + k: v for k, v in sd.items()}, x=x.numpy(), t=t.numpy(), y=y.numpy())
print("exported", len(sd), "tensors;", "trained" if a.weights else "random", "weights")
