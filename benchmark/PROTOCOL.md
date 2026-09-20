# M3 — RunsOnAnything Benchmark Protocol

## 1. Purpose

This document defines the reproducible benchmark protocol for M3
(RunsOnAnything).

The goal is to compare inference performance across deployment
backends while keeping the model, input, workload, and measurement
procedure consistent.

---

## 2. Target Environment

The initial benchmark target is a CPU-only Windows machine.

Environment captured during setup:

- OS: Windows
- Python: 3.11.9
- CPU: Intel 12th Gen Core i5-1235U
- Physical CPU cores: 10
- Logical CPU threads: 12
- RAM: 7.73 GB
- PyTorch: 2.14.0+cpu
- CUDA: unavailable

Performance results must be measured on the actual target machine.
No hardware performance numbers will be assumed.

---

## 3. Baseline Model

Model:

- YOLO26n
- Task: object detection
- Framework: PyTorch
- Execution device: CPU

The exact model file and model version must be recorded in the
benchmark result.

Model weights must not be committed to Git.

---

## 4. Input Configuration

Initial benchmark input:

- Image size: 640 × 640
- Batch size: 1
- Precision: FP32
- Device: CPU

The same input configuration must be used across the compared
backends whenever technically supported.

---

## 5. Benchmark Stages

The benchmark will progress through the following stages:

1. PyTorch baseline
2. ONNX export
3. ONNX Runtime inference
4. OpenVINO inference
5. Optional quantized inference
6. Final comparison

Each stage must record its own configuration and measured results.

---

## 6. Warm-up

Warm-up inference runs must be excluded from reported performance.

Purpose of warm-up:

- initialize the runtime
- load model execution structures
- avoid measuring one-time initialization overhead

The benchmark implementation will perform a fixed number of warm-up
iterations before timed inference.

---

## 7. Timed Inference

Only inference execution will be measured for the primary latency
metric.

The benchmark must distinguish between:

- model loading time
- preprocessing time
- inference time
- postprocessing time
- total end-to-end time

The primary backend comparison will use inference latency.

---

## 8. Repeated Measurements

Inference must be executed repeatedly rather than using a single
measurement.

The benchmark should report:

- number of timed runs
- mean latency
- median latency
- minimum latency
- maximum latency
- standard deviation
- throughput

Throughput will be reported as:

FPS = 1000 / mean_latency_ms

for batch size 1.

---

## 9. Memory Measurement

System memory usage should be recorded before and during benchmark
execution where practical.

Memory measurements must be clearly identified as system/process
measurements and must not be presented as GPU VRAM measurements on
this CPU-only system.

---

## 10. Reproducibility

Every benchmark result must record enough information to reproduce
the measurement.

At minimum:

- benchmark version
- timestamp
- operating system
- Python version
- CPU
- physical CPU cores
- logical CPU threads
- RAM
- framework/runtime
- framework/runtime version
- model
- model version or file
- input size
- batch size
- precision
- device
- warm-up runs
- timed runs
- measured metrics

---

## 11. Fair Comparison Rules

When comparing backends:

- Use the same model architecture.
- Use the same model weights.
- Use the same input size.
- Use the same batch size.
- Use the same precision where supported.
- Use the same test input.
- Exclude model-loading time from primary inference latency.
- Perform warm-up before timing.
- Record runtime versions.
- Do not mix measurements from different hardware environments.

If a backend requires a different configuration, the difference must be
documented explicitly.

---

## 12. Benchmark Output

Results will be stored under:

benchmark/results/

Machine-readable results:

- JSON
- CSV

Generated benchmark results must remain ignored by Git unless a
specific result is intentionally promoted as a project artifact.

---

## 13. Result Interpretation

The benchmark is intended to measure engineering trade-offs between
deployment approaches.

Results must be reported as measurements rather than assumptions.

The project will consider:

- latency
- throughput
- memory usage
- model compatibility
- deployment complexity
- reproducibility

No backend will be declared superior using a single metric alone.

---

## 14. Benchmark Integrity

Official benchmark results must not be collected while the system is
under abnormal memory pressure or during uncontrolled background
activity where that condition materially affects the measurement.

If a run is affected by an environmental problem, it must be recorded
as a failed or non-official run rather than silently used as a final
result.