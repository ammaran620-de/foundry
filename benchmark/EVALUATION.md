# Foundry M3 — Evaluation Specification

## Purpose

Define the evaluation protocol used to compare YOLO26n deployment
configurations throughout M3.

The goal is to measure the trade-off between model quality,
inference performance, resource usage, and deployment constraints.

---

## 1. Model

Primary model:

- YOLO26n object detector
- Pretrained checkpoint: `models/yolo26n.pt`
- Input resolution: 640x640
- Batch size: 1

The same model weights must be used as the reference model throughout
the deployment optimization ladder unless an experiment explicitly
documents a model change.

---

## 2. Deployment Ladder

The M3 optimization sequence is:

1. PyTorch CPU FP32
2. ONNX CPU
3. ONNX Runtime CPU
4. OpenVINO FP16 CPU
5. OpenVINO INT8 CPU
6. OpenVINO INT8 Intel Iris Xe GPU
7. ONNX Runtime WebGPU

Each stage must be measured rather than inferred from another stage.

---

## 3. Accuracy Evaluation

Primary detection metric:

- mAP50-95

Additional metrics should be recorded when useful for diagnosing
optimization regressions:

- precision
- recall
- per-class AP
- small-object performance

Accuracy evaluation must use ground-truth annotations.

The evaluation dataset must remain fixed across FP32, FP16,
and INT8 comparisons.

---

## 4. Calibration Dataset

INT8 calibration requires a representative and stratified image set.

Calibration images must be selected independently from the final
accuracy evaluation images.

The calibration selection should represent the operating distribution,
including factors such as:

- object scale
- object count
- class representation
- scene complexity
- lighting or visual variation when available

The calibration selection procedure must be deterministic and recorded.

Calibration images must not simply be selected as the first N files
in a directory.

---

## 5. Latency Metrics

Every runtime must report:

- p50 latency
- p95 latency
- mean latency
- minimum latency
- maximum latency
- standard deviation
- FPS

Timed inference must exclude:

- model loading
- model compilation
- one-time initialization

Preprocessing and postprocessing must be measured separately so that
pipeline bottlenecks can be identified.

---

## 6. Resource Metrics

Each deployment configuration should record:

- peak RAM usage
- model artifact size
- CPU thread configuration
- accelerator/device
- sustained throughput

Where practical, power-state conditions should also be recorded.

---

## 7. Sustained Benchmark

Short benchmark results are not sufficient.

M3 must include:

- cold-start measurement
- sustained measurement after 10 minutes

The sustained benchmark should be run under controlled conditions.

Power source should be recorded:

- AC power
- battery

when practical.

---

## 8. Thread Scaling

CPU runtime behavior must be evaluated across multiple thread
configurations.

The experiment should determine whether additional threads continue
to improve throughput or introduce diminishing returns.

The selected production configuration must be based on measurements.

---

## 9. Pipeline Profiling

The complete inference pipeline should be decomposed into measurable
stages where applicable:

- image decode
- resize
- letterbox
- color conversion
- normalization
- inference
- postprocessing
- NMS
- rendering

The objective is to identify the actual bottleneck rather than
assuming neural-network inference is always the dominant cost.

---

## 10. Fair Comparison Rules

All comparable runtime measurements must use:

- the same model version
- the same input resolution
- batch size 1
- the same evaluation input
- the same benchmark protocol
- the same warm-up policy
- the same timed-run count

Runtime-specific configuration must be explicitly recorded.

No individual measurements may be removed simply because they are
unfavorable.

Outliers must remain visible and be investigated when they materially
affect interpretation.

---

## 11. Accuracy / Performance Trade-off

Every optimization stage must be evaluated against the previous
reference configuration.

The final analysis must identify:

- latency change
- throughput change
- memory change
- model-size change
- accuracy change

The analysis must explicitly document at least one measurable
regression or trade-off when one occurs.

---

## 12. Pareto Analysis

The final M3 report must provide an accuracy-versus-latency Pareto
plot.

Every deployment configuration must be represented and labeled.

---

## 13. Classical CV Baseline

M3 also requires at least one classical computer-vision baseline
for an appropriate constrained vision problem.

Candidate methods include:

- background subtraction
- morphology
- optical flow
- template matching

The objective is to establish where a classical method may be
preferable to a neural detector.

---

## 14. Browser Deployment

The final M3 browser demonstration must support:

- webcam input
- local inference
- WebGPU
- WebAssembly fallback
- FPS display
- model download-size reporting

The demonstration must make clear that inference runs locally.

---

## 15. Reproducibility

Every benchmark result must record:

- timestamp
- hardware
- operating system
- Python version
- runtime version
- model
- input resolution
- batch size
- device
- thread configuration
- warm-up count
- timed-run count

Benchmark results must be reproducible from version-controlled code.

---

## 16. Cost Analysis

M3 must eventually estimate deployment cost using measured
performance and explicitly stated assumptions.

Required comparisons include:

- cost per 1,000 images
- cost per video-hour

Cloud GPU comparisons must state the source and assumptions used.

---

## 17. Failure Evidence

M3 must accumulate real failures encountered during implementation.

Each failure record should contain:

1. Problem
2. Reproduction
3. Observation
4. Root cause
5. Fix
6. Before/after measurement
7. Engineering lesson

The goal is to preserve evidence of real engineering work rather
than only documenting successful runs.

---

## 18. Final Results Table

The final M3 results table must contain:

| Configuration | mAP | p50 ms | p95 ms | FPS | RAM | Sustained 10 min |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch CPU FP32 | | | | | | |
| ONNX Runtime CPU | | | | | | |
| OpenVINO FP16 CPU | | | | | | |
| OpenVINO INT8 CPU | | | | | | |
| OpenVINO INT8 Iris Xe | | | | | | |
| Browser WebGPU | | | | | | |

---

## 19. Decision Record

For every major optimization, document:

- What changed?
- Why was it changed?
- What hypothesis did the change test?
- What improved?
- What became worse?
- What evidence supports the conclusion?

---

## Status

This file defines the M3 evaluation contract.

No INT8 calibration result is considered final until the calibration
dataset, evaluation dataset, metrics, and selection methodology are
documented.