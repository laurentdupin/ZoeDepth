# ZoeDepth native validation

The first correctness target is the default ZoeD-M12-N checkpoint used by
InferBridge.

## Canonical model boundary

- Source revision: `d87f17b2f5fdcb174cf4fb115491f4a6c60de152`
- Canonical file: `ZoeD_M12_N.pt`
- Canonical bytes: 1,443,406,099
- Canonical SHA-256:
  `c97f94c4d53c5b788af46c5da0462262aebb37ea116fd70014bcbba93146c33b`

The canonical file is a PyTorch archive containing model, optimizer, and epoch
objects. The dependency-free native runtime must not parse arbitrary pickle.
The development-only `zoedepth-export-pytorch-v1` converter extracts only
contiguous FP32 inference tensors. BEiT relative-position index tables and the
log-binomial class index are deterministic graph constants and are regenerated
by native code.

The derived representation is keyed by canonical SHA-256, converter ID, format
version, and ZoeDepth variant. It is a hidden cache artifact associated with
the shared canonical checkpoint, not a second catalog model or download.

- Inference FP32 tensors: 510
- Regenerated integer index tensors: 25
- Derived native bytes: 1,379,365,440
- Avoided optimizer/index/archive bytes: 64,040,659

The dependency-free memory-mapped native reader validates the file header,
variant, tensor directory bounds, ranks, dimensions, non-overlap, payload byte
counts, and derivation metadata. The real catalog checkpoint passes its native
model probe.

## BEiT-L/16 encoder gate

The correctness-first scalar implementation covers patch embedding, all 24
transformer blocks, per-block interpolated relative-position bias, attention,
LayerScale, and MLP branches. A deterministic 32x32 tensor was compared at the
four MiDaS feature taps:

| Block | Relative L1 | Maximum absolute error |
|---:|---:|---:|
| 5 | `1.80879e-6` (`0.000181%`) | `0.000100136` |
| 11 | `1.73310e-6` (`0.000173%`) | `0.000167847` |
| 17 | `1.78015e-6` (`0.000178%`) | `0.000343323` |
| 23 | `3.99074e-6` (`0.000399%`) | `0.00128174` |

## MiDaS decoder gate

The native graph now includes readout projection, multi-scale token
reassembly, transpose/stride resizing, scratch projections, all four feature
fusion stages, and the relative-depth head. At 32x32, the decoder taps have
relative L1 between `9.84871e-7` and `2.73322e-6`; the largest absolute
difference is `0.0644531` on an activation range reaching `25,683.7`.

## Full ZoeD-M12-N graph

The metric path includes the bottleneck projection, unbounded seed bins, four
projector/attractor stages, conditional log-binomial distribution, and
metric-depth expectation. The exported dependency-free DLL accepts planar
RGB FP32 in `[0,1]` and returns same-size metric depth.

| Input | Relative L1 | Maximum absolute error |
|---:|---:|---:|
| 32x32 | `0.00287806` (`0.287806%`) | `0.00276148` |
| 64x64 | `0.00308457` (`0.308457%`) | `0.00428212` |

These differences are accumulated FP32 execution-order drift and remain within
the required 1% bound.

## ZoeD-K and ZoeD-NK

The same dependency-free graph supports the indoor/outdoor ZoeD-K checkpoint
and the routed dual-head ZoeD-NK checkpoint. NK routing executes its
transformer on Vulkan and transfers only the two domain logits to select the
checkpoint's `nyu` or `kitti` head.

At 32x32 and 64x64, the scalar native oracle and Vulkan implementation produce
the same errors relative to the PyTorch CPU fixtures:

| Variant | Input | Relative L1 | Maximum absolute error |
|---|---:|---:|---:|
| N | 32x32 | `0.002875` | `0.002759` |
| N | 64x64 | `0.003084` | `0.004283` |
| K | 32x32 | `0.008139` | `0.007218` |
| K | 64x64 | `0.010085` | `0.011602` |
| NK | 32x32 | `0.003740` | `0.015990` |
| NK | 64x64 | `0.007511` | `0.018972` |

K at 64x64 is the sole fixture narrowly above 1% (`1.0085%`). The scalar CPU
oracle has the same `1.0094%` drift, while the Vulkan result differs from that
oracle by about `0.001%`; this is not a GPU-backend divergence.

## Vulkan full-graph gate

ABI version 2 adds `zoedepth_create_vulkan`, with an explicit physical-device
index and no silent CPU fallback. Vulkan executes the dynamic BEiT-L/16
encoder, MiDaS decoder, seed/projector/attractor metric head, conditional
log-binomial distribution, and final depth expectation.

All six variant/size fixtures above passed on each installed adapter:

- Radeon RX 9070 (`device_index=0`)
- GeForce GTX 1080 (`device_index=1`)
- Radeon RX 6700 XT (`device_index=2`)

The validation covered 18 full-graph executions. Cross-adapter differences
were below the existing scalar-oracle drift. Both the Vulkan-enabled and
CPU-only DLL builds pass the C ABI smoke test.

The current tensor ABI still prepares and uploads planar RGB on the host and
downloads depth for the caller. It does not advertise external-image import,
GPU-resident output leases, or zero-copy integration yet.

## Complete InferBridge image path

ABI version 3 adds `zoedepth_infer_bgra8_f32`. It reproduces the existing
InferBridge Python worker's `infer_pil()` defaults: BGRA-to-RGB conversion,
reflection padding, MiDaS minimal aspect-preserving resize (including
multiple-of-32 rounding), horizontal-flip augmentation, bicubic output resize,
and cropping back to the source dimensions. The selected `model_size` is the
worker's `Size` parameter. The existing already-prepared planar tensor entry
point remains unchanged.

A deterministic non-square 53x41 BGRA canary at network size 64 compared the
complete Python CPU image path with Vulkan on the RX 9070, GTX 1080, and RX
6700 XT. Mean relative error was `0.2806%`, maximum relative error `0.3052%`,
and maximum absolute error `0.003645` on all three. This is within the already
measured tensor-graph drift.
`native/tools/validate_image_path.py` reproduces the check from the canonical
checkpoint and its derived native representation.

## First performance pass

Decoder and metric operations now use bounded Vulkan command batches.
Graph-internal buffer snapshots are recorded in the active command buffer with
explicit transfer/compute barriers, avoiding a queue submission and fence wait
for every snapshot. The NK routing transformer is also one bounded submission;
its two-logit domain decision remains the only mid-graph host read.

On the Radeon RX 9070, the N 64x64 fixture improved from a `76.0 ms`
steady-state median to `63.8-66.1 ms` across three 21-iteration runs
(`13-16%`). All 18 variant/size/adapter correctness canaries retained the
results above.

## Embedded InferBridge harness

The native DLL exports `ibrh_get_api` for InferBridge harness ABI 1.0. The
single ZoeDepth catalog entry selects `ZoeN`, `ZoeK`, or `ZoeNK` through the
existing `Models` parameter and selects a multiple-of-32 network size through
`Size`. `model_path` remains the hidden content-addressed `.zoe` derivation
of the canonical checkpoint selected for that variant; it is not a duplicate
model card or canonical download.

The harness accepts one host-memory BGRA8 capture, executes the complete image
and metric graph, then reproduces the Python worker's final
min/max-to-255 normalization, inversion, and truncating uint8 conversion. It
returns a leased source-size `DEPTH_UNORM8` image and preserves
`source_frame_id` and timestamp. An acquired output remains valid after its
job handle is released.

Capability reporting advertises only host input/output and one synchronous
in-flight job. The selected Vulkan device executes the full graph; image
upload and metric-depth download remain host boundaries. External GPU
resources, asynchronous completion, and cancellation are not advertised.

The Windows Release ABI and full-graph harness tests pass for all three
derived ZoeN/ZoeK/ZoeNK models on the RX 9070. The deployed ZoeN 53x41,
size-64 uint8 output was also compared against Python CPU on every GPU:
maximum pixel deviation was one 8-bit level on all three adapters. The
underlying 18 variant/size/adapter metric gates remain as reported above.
