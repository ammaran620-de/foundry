# M3 — RunsOnAnything

CPU-first deployment benchmarking for YOLO26n across PyTorch, ONNX Runtime, OpenVINO, Intel iGPU, and local browser inference.

## What This Module Proves

M3 measures how the same inference workload behaves as the runtime and execution target change.

The work follows the deployment ladder:

`PyTorch CPU FP32 → ONNX Runtime CPU → OpenVINO FP16 CPU → OpenVINO INT8 CPU → OpenVINO INT8 Intel GPU → Browser WebGPU`

The benchmark records latency, throughput, accuracy, memory, thread behavior, sustained behavior, and failure modes.

---

## Final Benchmark Evidence

### Standardized Backend Comparison

| Runtime | Mean | p50 | p95 | FPS |
|---|---:|---:|---:|---:|
| PyTorch CPU | 277.20 ms | 266.82 ms | 332.43 ms | 3.61 |
| ONNX Runtime CPU | 69.08 ms | 66.45 ms | 79.96 ms | 14.48 |
| OpenVINO FP16 CPU | 120.91 ms | 117.37 ms | 137.99 ms | 8.27 |

All three configurations use the same 640×640 float32 input and batch size 1. The recorded comparison keeps the required NMS path inside the measured inference path.

### Deployment Accuracy Cross-Check

| Configuration | mAP50-95 | mAP50 | mAP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 reference | 0.4053 | 0.5629 | 0.4356 | 0.6610 | 0.5181 |
| ONNX Runtime CPU | 0.3468 | 0.4537 | 0.3763 | 0.6954 | 0.4915 |
| OpenVINO FP16 CPU | 0.3451 | 0.4514 | 0.3739 | 0.6729 | 0.4916 |

Accuracy is treated as a deployment property, not only a model-training property.

### Sustained INT8 — 10 Minutes

| Device | Threads | Mean | p50 | p95 | FPS | Peak RSS | Drift | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Intel i5-1235U CPU | 4 | 50.30 ms | 49.62 ms | 53.54 ms | 19.87 | 206.13 MB | +1.76% | 0 |
| Intel UHD Graphics iGPU | N/A | 16.16 ms | 15.79 ms | 17.65 ms | 61.78 | 287.41 MB | +10.76% | 0 |

The CPU sustained run completed 600 seconds with 11,921 successful runs and zero timeouts.

The Intel iGPU sustained run completed 600 seconds with 37,070 successful runs and zero timeouts.

The GPU result is faster, but its measured sustained latency drift is higher.

Full recorded evidence:
- `benchmark/results/backend_comparison.json`
- `benchmark/results/deployment_accuracy.json`
- `benchmark/results/sustained_int8_robust.json`
- `benchmark/results/M3_FINAL_TABLE.md`

---

## Thread Scaling

INT8 CPU thread scaling was measured rather than assuming more threads would be faster.

The recorded results show the best mean latency and throughput at 4 threads:

| Threads | Mean | p50 | p95 | FPS |
|---:|---:|---:|---:|---:|
| 1 | 53.37 ms | 49.56 ms | 74.49 ms | 18.74 |
| 2 | 52.63 ms | 50.06 ms | 67.69 ms | 19.00 |
| 4 | 52.46 ms | 51.16 ms | 58.70 ms | 19.06 |
| 6 | 62.16 ms | 61.33 ms | 67.69 ms | 16.09 |
| 8 | 57.09 ms | 56.84 ms | 60.22 ms | 17.52 |
| 10 | 58.51 ms | 58.11 ms | 62.41 ms | 17.09 |
| 12 requested / 10 actual | 61.30 ms | 60.77 ms | 67.91 ms | 16.31 |

---

## Pipeline Profile

The INT8 deployment was profiled by pipeline stage.

### CPU

| Stage | Mean |
|---|---:|
| Decode | 12.72 ms |
| Resize | 0.17 ms |
| Color conversion | 0.28 ms |
| Normalize | 7.73 ms |
| Inference | 58.93 ms |
| Postprocess | 0.00 ms |
| End-to-end | 79.92 ms |

Inference accounts for approximately 73.7% of the measured end-to-end time.

### Intel iGPU

| Stage | Mean |
|---|---:|
| Decode | 12.08 ms |
| Resize | 0.26 ms |
| Color conversion | 0.25 ms |
| Normalize | 7.17 ms |
| Inference | 17.65 ms |
| Postprocess | 0.00 ms |
| End-to-end | 37.48 ms |

The GPU shifts the pipeline bottleneck: inference becomes approximately 47.1% of the measured end-to-end time.

---

## Browser Deployment

The browser deployment runs locally with:

- YOLO26n
- ONNX model
- WebGPU execution
- WebAssembly fallback
- webcam input
- local inference
- displayed FPS and inference latency
- model download size

Browser model:

`browser/public/models/yolo26n_webgpu.onnx`

The model is exported without embedded NMS for the browser path.

Current browser measurement observed during validation was approximately 2 FPS with several hundred milliseconds of inference latency on this machine.

The browser demo therefore serves as deployment/runtime evidence rather than an arbitrary-object recognition benchmark.

---

## Failure Gallery

Observed browser/runtime failures are documented under:

`failures/m3/`

The failure inventory includes runtime failures and semantic detection failures observed during validation.

Examples include:

- slow WebGPU inference
- detached ArrayBuffer runtime failure
- missed COCO mouse detections
- incorrect classifications for objects outside the checkpoint's learned classes
- intermittent cell-phone/laptop behavior
- false-positive detections

Failures are retained as evidence rather than hidden.

See:

`failures/m3/README.md`

`failures/m3/INDEX.md`

---

## Accuracy and Failure Scope

The browser model is the standard YOLO26n COCO checkpoint.

It is not trained to recognize arbitrary user-defined objects such as every kind of pen, diary, charger, wire, or microphone.

Consequently, browser observations are interpreted in two separate dimensions:

1. Can the model execute locally in the target runtime?
2. How does the model actually behave on the supplied visual input?

A successful deployment does not imply perfect semantic recognition.

---

## 100× Scale Considerations

The measurements show that optimization changes where the bottleneck lives.

At the CPU baseline, inference dominates runtime.

After moving to optimized INT8 execution and especially the Intel iGPU, inference latency drops substantially, making image decode and preprocessing a larger fraction of end-to-end execution.

At larger deployment scale, this means the engineering problem is no longer only model inference. It also includes:

- frame decode cost
- preprocessing cost
- memory behavior
- sustained thermal behavior
- device utilization
- concurrency and thread configuration
- model distribution/download size
- browser/runtime limitations

The benchmark evidence is therefore intended to support scaling decisions from measured behavior rather than from peak single-run FPS.

---

## Reproducibility

Primary benchmark protocol:

`benchmark/PROTOCOL.md`

Accuracy specification:

`benchmark/EVALUATION.md`

Evaluation subset:

`benchmark/evaluation/coco1000/`

Calibration subset:

`benchmark/evaluation/`

Benchmark scripts:

`benchmark/`

Generated evidence:

`benchmark/results/`

The repository includes a regression gate to detect unacceptable accuracy or latency regressions.

---

## Current M3 Status

### Completed

- Reproducible environment capture
- Standardized backend comparison
- ONNX Runtime CPU benchmark
- OpenVINO FP16 CPU benchmark
- INT8 calibration and deployment
- INT8 thread scaling
- 10-minute sustained CPU benchmark
- 10-minute sustained Intel iGPU benchmark
- Pipeline profiling
- Browser WebGPU deployment
- Cross-platform CI validation
- Benchmark regression gate
- Observed failure inventory
- Final benchmark evidence table
- M3 deployment lifecycle diagram
- Browser deployment documentation
- 60-second proof video

### Publication Note

The engineering evidence, benchmark results, browser deployment, failure inventory, and supporting documentation are included in the repository.

A stable hosted browser demo link is not currently included; the browser deployment is provided as a reproducible local deployment.

---

## Repository Structure

```text
m3-runsonanything/
├── benchmark/
│   ├── evaluation/
│   ├── input/
│   ├── results/
│   ├── EVALUATION.md
│   └── PROTOCOL.md
│
├── browser/
│   ├── public/models/
│   ├── src/
│   └── README.md
│
├── docs/
│   └── M3_DEPLOYMENT_LIFECYCLE.md
│
├── failures/
│   └── m3/
│
├── models/
├── scripts/
├── src/
└── README.md