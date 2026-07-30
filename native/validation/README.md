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

The Zoe metric-bin head remains pending.
