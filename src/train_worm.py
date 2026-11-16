"""Additive wrapper around the native train.py: identical training, plus
  --seed S         re-seed Python/NumPy/Torch after safe_state (S=0 changes nothing);
  --run_record F   write wall time, peak VRAM and final Gaussian count to JSON F.
"""

import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

REPO = Path(os.environ["R2_GAUSSIAN_ROOT"])  # clone of R2-Gaussian with patches/ applied
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402  (unchanged native module)


def main():
    argv = sys.argv[1:]
    seed, record = 0, None
    if "--seed" in argv:
        i = argv.index("--seed"); seed = int(argv[i + 1]); del argv[i:i + 2]
    if "--run_record" in argv:
        i = argv.index("--run_record"); record = argv[i + 1]; del argv[i:i + 2]
    # --- same argument handling as train.py __main__ ---
    parser = ArgumentParser(description="Training script parameters")
    lp = T.ModelParams(parser)
    op = T.OptimizationParams(parser)
    pp = T.PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    T.safe_state(args.quiet)
    if seed != 0:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tb_writer = T.prepare_output_and_logger(args)
    if tb_writer is not None:  # no TensorBoard logs (~20 MB per run); all metrics are in eval/*.yml
        tb_writer.close()
        tb_writer = None
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    T.training(lp.extract(args), op.extract(args), pp.extract(args), tb_writer,
               args.test_iterations, args.save_iterations, args.checkpoint_iterations,
               args.start_checkpoint)
    elapsed = time.time() - t0
    if record:
        import pickle

        pc = os.path.join(args.model_path, f"point_cloud/iteration_{args.iterations}/point_cloud.pickle")
        n = int(pickle.load(open(pc, "rb"))["xyz"].shape[0]) if os.path.exists(pc) else None
        with open(record, "x") as f:
            json.dump({"seed": seed, "argv": argv, "elapsed_seconds": elapsed,
                       "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "final_gaussians": n, "gpu": torch.cuda.get_device_name(0)}, f, indent=1)


if __name__ == "__main__":
    main()
