# M3 Proof Pack

## Requirement → Evidence Map

| Foundry M3 requirement | Repository evidence | Status |
|---|---|---|
| Reproducible benchmark protocol | `benchmark/PROTOCOL.md` | Complete |
| Accuracy evaluation specification | `benchmark/EVALUATION.md` | Complete |
| PyTorch CPU FP32 baseline | `benchmark/results/pytorch_baseline.json` | Complete |
| ONNX Runtime CPU | `benchmark/results/onnxruntime_baseline.json` | Complete |
| OpenVINO FP16 CPU | `benchmark/results/openvino_baseline.json` | Complete |
| OpenVINO INT8 CPU | `benchmark/results/sustained_int8_robust.json` | Complete |
| OpenVINO INT8 Intel GPU | `benchmark/results/sustained_int8_robust.json` | Complete |
| Accuracy cross-check | `benchmark/results/deployment_accuracy.json` | Complete |
| CPU thread scaling | `benchmark/results/thread_scaling_int8.json` | Complete |
| 10-minute sustained run | `benchmark/results/sustained_int8_robust.json` | Complete |
| Pipeline profiling | `benchmark/results/pipeline_profile_int8.json` | Complete |
| Browser local inference | `browser/` | Complete |
| Browser WebGPU path | `browser/` | Complete |
| Browser WebAssembly fallback | `browser/` | Complete |
| Failure documentation | `failures/m3/` | Complete as text records |
| Final benchmark table | `benchmark/results/M3_FINAL_TABLE.md` | Complete |
| README evidence summary | `README.md` | Complete |
| Cross-platform validation | `.github/workflows/` | Complete |
| Regression gate | `benchmark/regression_gate.py` | Complete |
| Final failure screenshots/evidence frames | No saved artifacts | Missing |
| 60-second proof video | No saved artifact | Missing |
| Stable public browser URL | Not established | Missing |
| Public technical write-up | Not established | Missing |

## Core Recorded Result

The standardized comparison and sustained INT8 measurements are documented in:

`benchmark/results/M3_FINAL_TABLE.md`

## Failure Evidence Policy

Browser failures without saved image/frame artifacts are documented as observations only.
No timestamps, screenshots, or visual evidence are fabricated.

Quantitative failures and tradeoffs are backed by the recorded benchmark JSON files.

## Current State

The engineering benchmark/evidence portion of M3 is substantially documented.

The remaining items are publication/proof artifacts rather than another model-training or backend-optimization loop.
