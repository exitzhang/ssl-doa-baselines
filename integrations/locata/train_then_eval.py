#!/usr/bin/env python
"""Foreground train -> validation lock -> DICIT eval wrapper for gpu_queue.

Training and evaluation share one GPU slot, mirroring LOCATA's
``train_then_sweep.py``::

    python integrations/locata/train_then_eval.py \
      --workdir /path/to/ssl-doa-baselines \
      --config integrations/locata/configs/tf_mamba_dicit.toml \
      --data-config integrations/locata/configs/data_local.toml \
      --output runs/tf_mamba_spectrum_s42 --seed 42 --device-index 0

The script returns 0 only when training produced ``checkpoint.pth`` and
``validation.json`` and the DICIT evaluation wrote ``eval_dicit.json``.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd):
    print("[train_then_eval] $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--locata-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-dicit", action="store_true")
    args = p.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    runner = workdir / "integrations" / "locata" / "run.py"
    common = ["--locata-root", args.locata_root, "--config", args.config,
              "--data-config", args.data_config,
              "--device-index", str(args.device_index)]
    if args.seed is not None:
        common += ["--seed", str(args.seed)]

    train_extra = []
    if args.epochs is not None:
        train_extra += ["--epochs", str(args.epochs)]
    if args.max_train_batches is not None:
        train_extra += ["--max-train-batches", str(args.max_train_batches)]
    rc = run([args.python, str(runner), *common, "--output", args.output,
              "train", *train_extra], workdir)
    if rc != 0:
        return rc

    out = Path(args.output)
    ckpt = out / "checkpoint.pth"
    validation = out / "validation.json"
    if not ckpt.is_file() or not validation.is_file():
        print(f"[train_then_eval] missing {ckpt} or {validation}", file=sys.stderr)
        return 1
    if args.skip_dicit:
        return 0

    rc = run([args.python, str(runner), *common, "--output",
              str(out / "eval_dicit.json"), "eval",
              "--checkpoint", str(ckpt), "--split", "dicit",
              "--validation", str(validation)], workdir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
