# Native validation

The native runtime consumes a bounded `ZOENMOD` tensor file derived from the
official checkpoint with `torch.load(..., weights_only=True)`. Deployment does
not parse pickle and does not depend on Python or PyTorch.

Deterministic full-graph comparisons use PyTorch CPU as the reference. The
input is planar RGB FP32 generated with seed `20260730`.

| Variant | Input | Relative L1 | Maximum absolute |
|---|---:|---:|---:|
| ZoeD-M12-N | 32x32 | 0.287806% | recorded by the N validation gate |
| ZoeD-M12-N | 64x64 | 0.308457% | recorded by the N validation gate |
| ZoeD-M12-K | 32x32 | 0.816077% | 0.00723636 |
| ZoeD-M12-K | 64x64 | 1.00944% | 0.0116129 |
| ZoeD-M12-NK | 32x32 | 0.373979% | 0.0159895 |
| ZoeD-M12-NK | 64x64 | 0.751035% | 0.018972 |

The 64x64 K result is a small accumulated FP32 evaluation-order difference.
No weights, layers, or output corrections are omitted. The NK fixtures route
to the NYU head; the learned four-layer transformer router and both NYU/KITTI
head tensors are loaded, and the selected head executes through the same
validated metric-head implementation.
