"""Validate the native BGRA image API against ZoeDepth infer_pil()."""

from __future__ import annotations

import argparse
import ctypes
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
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--width", type=int, default=53)
    parser.add_argument("--height", type=int, default=41)
    args = parser.parse_args()

    rng = np.random.default_rng(20260730)
    bgra = rng.integers(
        0, 256, (args.height, args.width, 4), dtype=np.uint8)
    bgra[:, :, 3] = 255
    rgb = bgra[:, :, [2, 1, 0]]

    # The validation environment intentionally uses CPU-only PyTorch. Some
    # torchvision wheels still register the NMS fake kernel unconditionally;
    # declaring its schema is enough for importing transforms (NMS is unused).
    torchvision_library = torch.library.Library("torchvision", "DEF")
    torchvision_library.define(
        "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    model = build_model(
        args.repo.resolve(), args.midas_repo.resolve(),
        args.checkpoint.resolve(), args.size, "n")
    with torch.inference_mode():
        reference = model.infer_pil(rgb).astype(np.float32)

    library = ctypes.CDLL(str(args.dll.resolve()))
    context = ctypes.c_void_p()
    library.zoedepth_create_vulkan.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p)]
    library.zoedepth_create_vulkan.restype = ctypes.c_int
    library.zoedepth_infer_bgra8_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64]
    library.zoedepth_infer_bgra8_f32.restype = ctypes.c_int
    library.zoedepth_last_error.restype = ctypes.c_char_p
    library.zoedepth_destroy.argtypes = [ctypes.c_void_p]

    status = library.zoedepth_create_vulkan(
        str(args.native_model.resolve()).encode(), 0, args.device,
        ctypes.byref(context))
    if status:
        raise RuntimeError(library.zoedepth_last_error().decode())
    actual = np.empty((args.height, args.width), dtype=np.float32)
    try:
        status = library.zoedepth_infer_bgra8_f32(
            context,
            bgra.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bgra.strides[0], args.width, args.height, args.size,
            actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            actual.size)
        if status:
            raise RuntimeError(library.zoedepth_last_error().decode())
    finally:
        library.zoedepth_destroy(context)

    difference = np.abs(actual - reference)
    relative = difference / np.maximum(np.abs(reference), 1e-6)
    report = {
        "shape": list(reference.shape),
        "reference_minimum": float(reference.min()),
        "reference_maximum": float(reference.max()),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "maximum_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
    }
    print(json.dumps(report, indent=2))
    if not np.isfinite(actual).all() or relative.mean() > 0.02:
        raise SystemExit("native image-path accuracy gate failed")


if __name__ == "__main__":
    main()
