# M3 Final Benchmark Evidence

## Standardized Backend Comparison

| Runtime | Mean Latency | p50 | p95 | FPS |
|---|---:|---:|---:|---:|
| PyTorch CPU | 277.20 ms | 266.82 ms | 332.43 ms | 3.61 |
| ONNX Runtime CPU | 69.08 ms | 66.45 ms | 79.96 ms | 14.48 |
| OpenVINO FP16 CPU | 120.91 ms | 117.37 ms | 137.99 ms | 8.27 |

## Deployment Accuracy Cross-Check

| Configuration | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 reference | 0.4053 | 0.5629 | 0.4356 | 0.6610 | 0.5181 |
| ONNX Runtime CPU | 0.3468 | 0.4537 | 0.3763 | 0.6954 | 0.4915 |
| OpenVINO FP16 CPU | 0.3451 | 0.4514 | 0.3739 | 0.6729 | 0.4916 |

## Sustained INT8 — 10 Minutes

| Device | Threads | Mean | p50 | p95 | FPS | Peak RSS | Drift | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Intel i5-1235U CPU | 4 | 50.30 ms | 49.62 ms | 53.54 ms | 19.87 | 206.13 MB | +1.76% | 0 |
| Intel UHD Graphics iGPU | N/A | 16.16 ms | 15.79 ms | 17.65 ms | 61.78 | 287.41 MB | +10.76% | 0 |

## What Got Worse?

The sustained CPU run remained stable, with only +1.76% latency drift.

The Intel iGPU achieved substantially lower latency, but sustained latency drift reached +10.76%, so the GPU result is not described as perfectly thermally/stability-neutral.

## Measurement Notes

All values are taken from recorded benchmark result files.
No benchmark was rerun for this table.

