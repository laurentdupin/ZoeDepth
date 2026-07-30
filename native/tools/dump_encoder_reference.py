"""Generate deterministic ZoeDepth BEiT encoder fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def build_model(
        repo: Path, midas_repo: Path, checkpoint: Path, size: int,
        variant: str = "n"):
    sys.path.insert(0, str(repo))
    from zoedepth.models.base_models.midas import MidasCore
    from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth

    midas = torch.hub.load(
        str(midas_repo), "DPT_BEiT_L_384",
        pretrained=False, source="local")
    # timm >=0.9 split stochastic-depth modules. Evaluation makes the two
    # forms identical; this alias preserves the pinned MiDaS forward adapter.
    for block in midas.pretrained.model.blocks:
        if not hasattr(block, "drop_path"):
            block.drop_path = block.drop_path1
    core = MidasCore(
        midas, trainable=False, fetch_features=True, freeze_bn=True,
        img_size=[size, size], keep_aspect_ratio=True)
    core.set_output_channels("DPT_BEiT_L_384")
    if variant in ("n", "k"):
        model = ZoeDepth(
            core, n_bins=64, bin_embedding_dim=128,
            bin_centers_type=(
                "normed" if variant == "k" else "softplus"),
            n_attractors=[16, 8, 4, 1],
            attractor_alpha=1000, attractor_gamma=2,
            attractor_kind="mean", attractor_type="inv",
            min_temp=0.0212, max_temp=50.0,
            memory_efficient=True)
    elif variant == "nk":
        from zoedepth.models.zoedepth_nk.zoedepth_nk_v1 import (
            ZoeDepthNK)
        from zoedepth.utils.easydict import EasyDict
        model = ZoeDepthNK(
            core,
            bin_conf=[
                EasyDict({"name": "nyu", "n_bins": 64,
                          "min_depth": 1e-3, "max_depth": 10.0}),
                EasyDict({"name": "kitti", "n_bins": 64,
                          "min_depth": 1e-3, "max_depth": 80.0}),
            ],
            bin_embedding_dim=128, bin_centers_type="softplus",
            n_attractors=[16, 8, 4, 1],
            attractor_alpha=1000, attractor_gamma=2,
            attractor_kind="mean", attractor_type="inv",
            min_temp=0.0212, max_temp=50.0,
            memory_efficient=True)
    else:
        raise ValueError(f"unsupported ZoeDepth variant: {variant}")
    archive = torch.load(
        checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(archive["model"], strict=False)
    if incompatible.missing_keys or any(
            not name.endswith(".relative_position_index")
            for name in incompatible.unexpected_keys):
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model.eval()
    return model


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

    captures = {}
    handles = []
    for block_index in (5, 11, 17, 23):
        handles.append(
            model.core.core.pretrained.model.blocks[
                block_index].register_forward_hook(
                    lambda _module, _input, output, index=block_index:
                    captures.__setitem__(index, output.detach().cpu())))

    generator = torch.Generator().manual_seed(20260730)
    rgb = torch.rand(
        1, 3, args.size, args.size, generator=generator)
    prepared = model.core.prep(rgb)
    with torch.inference_mode():
        model.core.core(prepared)
    for handle in handles:
        handle.remove()

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prepared.numpy().astype(np.float32).tofile(
        prefix.with_suffix(".input.bin"))
    report = {"input_shape": list(prepared.shape), "captures": []}
    for ordinal, block_index in enumerate((5, 11, 17, 23)):
        tensor = captures[block_index]
        tensor.numpy().astype(np.float32).tofile(
            prefix.parent / f"{prefix.name}.block{block_index}.bin")
        report["captures"].append({
            "block": block_index,
            "shape": list(tensor.shape),
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
            "mean": float(tensor.mean()),
            "sum": float(tensor.double().sum()),
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
