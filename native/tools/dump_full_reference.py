"""Generate deterministic full ZoeDepth fixtures and decoder taps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dump_encoder_reference import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--midas-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--variant", choices=("n", "k", "nk"), default="n")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    model = build_model(
        args.repo.resolve(), args.midas_repo.resolve(),
        args.checkpoint.resolve(), args.size, args.variant)

    generator = torch.Generator().manual_seed(20260730)
    rgb = torch.rand(
        1, 3, args.size, args.size, generator=generator)
    relative = {}
    handle = model.core.core.register_forward_hook(
        lambda _module, _input, output:
        relative.__setitem__("depth", output.detach().cpu()))
    with torch.inference_mode():
        outputs = model(
            rgb, return_final_centers=True, return_probs=True)
        depth = outputs["metric_depth"].cpu()
    handle.remove()

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".rgb.bin"))
    relative["depth"].numpy().astype(np.float32).tofile(
        prefix.with_suffix(".relative.bin"))
    depth.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".depth.bin"))
    outputs["bin_centers"].cpu().numpy().astype(np.float32).tofile(
        prefix.with_suffix(".centers.bin"))
    outputs["probs"].cpu().numpy().astype(np.float32).tofile(
        prefix.with_suffix(".probs.bin"))
    names = ("out_conv", "l4_rn", "r4", "r3", "r2", "r1")
    report = {
        "variant": args.variant,
        "rgb_shape": list(rgb.shape),
        "relative_shape": list(relative["depth"].shape),
        "depth_shape": list(depth.shape),
        "depth_minimum": float(depth.min()),
        "depth_maximum": float(depth.max()),
        "depth_mean": float(depth.mean()),
        "depth_sum": float(depth.double().sum()),
        "decoder_taps": [],
    }
    if "domain_logits" in outputs:
        logits = outputs["domain_logits"].detach().cpu()
        logits.numpy().astype(np.float32).tofile(
            prefix.with_suffix(".domain_logits.bin"))
        report["domain_logits"] = logits.flatten().tolist()
        report["selected_domain"] = (
            "nyu" if int(logits.sum(dim=0).argmax()) == 0
            else "kitti")
    for name in names:
        tensor = model.core.core_out[name].detach().cpu()
        tensor.numpy().astype(np.float32).tofile(
            prefix.parent / f"{prefix.name}.{name}.bin")
        report["decoder_taps"].append({
            "name": name,
            "shape": list(tensor.shape),
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
            "mean": float(tensor.mean()),
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
