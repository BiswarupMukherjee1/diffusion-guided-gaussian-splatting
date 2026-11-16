"""Build the nine test cases (poses 4, 1031, 8916; mid slab; arcs 120, 90, 60 degrees from 0) and their
starting points: dataset in the R2-Gaussian format, native FBP/FDK start (initialize_pcd.py of
R2-Gaussian, unchanged settings) and the SIRT start. Phantoms must exist (build_phantom.py).

  python prepare_cases.py            # uses $LACT_WORK/phantoms and writes $LACT_WORK/cases
"""

import subprocess
import sys
from pathlib import Path

import core as C
import make_inits
from prepare_case import build

INIT_ARGS = ["--recon_method", "fdk", "--n_points", "50000", "--density_thresh", "0.05", "--density_rescale", "0.15"]


def main():
    root = C.RUNS_WORM
    for w in C.TEST_POSES:
        for span in C.SPANS:
            cdir, _ = build("A", w, "mid", span, 0, 0, 0, root / "phantoms", root / "cases")
            if not (cdir / "init.npy").exists():
                subprocess.run([sys.executable, "-u", "initialize_pcd.py", "--data", cdir, "--output", cdir / "init.npy"]
                               + INIT_ARGS, cwd=C.R2_ROOT, check=True, stdout=subprocess.DEVNULL)
            if not (cdir / "init_fdk_diag.json").exists():
                subprocess.run([sys.executable, "-u", Path(__file__).with_name("make_inits.py"), cdir, "--fdk-diag"],
                               check=True)
            make_inits.build_case(cdir, ["sirt"])


if __name__ == "__main__":
    main()
