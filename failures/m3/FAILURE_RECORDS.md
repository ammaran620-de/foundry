# M3 Failure Records

These records document observed or quantitatively measured M3 failures and tradeoffs.
No synthetic timestamps or fabricated screenshots are included.

## F-001 — WebGPU Embedded-NMS Latency

**Layer:** Browser / model graph  
**Observation:** The first browser model used embedded NMS.  
**Measured behavior:** Approximately 547.7 ms inference latency with no useful real-time FPS.  
**Impact:** Browser execution was too slow for practical interactive inference.  
**Action:** Re-exported the browser model without embedded NMS.  
**Status:** Mitigated by changing the browser deployment artifact.

---

## F-002 — WebGPU NMS-Free Latency

**Layer:** Browser / runtime  
**Observation:** Removing embedded NMS did not make the model real-time on this machine.  
**Measured behavior:** Approximately 462–503 ms inference and about 1.46–1.67 FPS during validation.  
**Impact:** Browser inference remained latency-bound.  
**Status:** Accepted deployment limitation and retained as M3 evidence.

---

## F-003 — Detached ArrayBuffer Runtime Failure

**Layer:** Browser / Web Worker tensor lifecycle  
**Observation:** Browser inference encountered a detached ArrayBuffer / Worker failure.  
**Impact:** Inference could terminate instead of producing a result.  
**Action:** Tensor/input ownership was changed so each inference uses fresh buffers/tensors.  
**Status:** Fixed.

---

## F-004 — COCO Mouse Miss

**Layer:** Model semantic detection  
**Observation:** A mouse was present in browser validation but was not reliably detected.  
**Impact:** False negative on a class that exists in the COCO checkpoint.  
**Status:** Retained as a model failure example.

---

## F-005 — Diary Misclassification

**Layer:** Model semantic detection  
**Observation:** A diary was mapped to an incorrect COCO class.  
**Impact:** False semantic classification.  
**Scope:** Diary is not a dedicated class in the standard COCO checkpoint.  
**Status:** Retained as an expected out-of-scope/generalization failure.

---

## F-006 — Pen / Highlighter Misclassification

**Layer:** Model semantic detection  
**Observation:** A pen/highlighter was assigned an incorrect class.  
**Impact:** False semantic classification.  
**Scope:** Pen/highlighter is not a dedicated class in the standard COCO checkpoint.  
**Status:** Retained as failure evidence.

---

## F-007 — Pen / Highlighter Miss

**Layer:** Model semantic detection  
**Observation:** Pen/highlighter views also produced no reliable detection.  
**Impact:** False negative.  
**Status:** Retained as failure evidence.

---

## F-008 — Microphone Miss

**Layer:** Model semantic detection  
**Observation:** A microphone was not reliably detected.  
**Impact:** False negative.  
**Scope:** Microphone is not a dedicated COCO checkpoint class.  
**Status:** Retained as failure evidence.

---

## F-009 — False-Positive Detections

**Layer:** Model semantic detection / confidence filtering  
**Observation:** Hand/object regions produced detections that did not correspond reliably to the intended object.  
**Impact:** False positives in the browser UI.  
**Status:** Retained as failure evidence.

---

## F-010 — Intermittent Cell-Phone Detection

**Layer:** Model semantic detection  
**Observation:** Cell-phone detection was correct in some views but unreliable in others.  
**Impact:** Inconsistent temporal detection.  
**Status:** Retained as failure evidence.

---

## F-011 — Laptop/Object Confusion

**Layer:** Model semantic detection  
**Observation:** Laptop-related detections were inconsistent across views.  
**Impact:** Semantic instability.  
**Status:** Retained as failure evidence.

---

## F-012 — CPU Thread Scaling Regression

**Layer:** OpenVINO CPU execution  
**Observation:** Increasing thread count beyond 4 reduced throughput rather than increasing it.  
**Measured behavior:**
- 4 threads: 52.46 ms mean, 19.06 FPS
- 6 threads: 62.16 ms mean, 16.09 FPS
- 10 threads: 58.51 ms mean, 17.09 FPS
- 12 requested / 10 actual: 61.30 ms mean, 16.31 FPS  
**Impact:** More CPU parallelism increased latency on this workload.  
**Status:** Measured and retained.

---

## F-013 — Single-Thread Tail-Latency Regression

**Layer:** OpenVINO CPU execution  
**Observation:** One thread produced a lower p50 than the 4-thread configuration but a much worse tail.  
**Measured behavior:** 49.56 ms p50 versus 74.49 ms p95 and 89.66 ms maximum latency.  
**Impact:** Low median latency did not imply stable tail behavior.  
**Status:** Measured and retained.

---

## F-014 — Requested Thread Count Exceeded Actual Runtime Capacity

**Layer:** OpenVINO CPU execution  
**Observation:** Requesting 12 threads resulted in 10 actual threads.  
**Measured behavior:** `requested_threads=12`, `actual_threads=10`.  
**Impact:** Configuration intent and runtime execution differed.  
**Status:** Measured and retained.

---

## F-015 — INT8 Deployment Accuracy Loss

**Layer:** Quantized deployment  
**Observation:** The INT8 deployment did not preserve the PyTorch FP32 reference accuracy.  
**Measured behavior:** PyTorch FP32 reference mAP50-95 = 0.4053; recorded INT8 validation was approximately 0.337.  
**Impact:** Quantization introduced a measurable accuracy tradeoff.  
**Status:** Retained as a quantified optimization tradeoff.

---

## F-016 — OpenVINO FP16 Slower Than ONNX Runtime CPU

**Layer:** Runtime/backend selection  
**Observation:** OpenVINO FP16 was not faster than ONNX Runtime CPU in the standardized benchmark.  
**Measured behavior:**
- ONNX Runtime CPU: 69.08 ms mean, 14.48 FPS
- OpenVINO FP16 CPU: 120.91 ms mean, 8.27 FPS  
**Impact:** A different optimized runtime did not automatically provide lower latency.  
**Status:** Measured and retained.

---

## F-017 — Sustained Intel iGPU Latency Drift

**Layer:** Sustained GPU execution  
**Observation:** The Intel iGPU sustained benchmark showed measurable latency drift over 10 minutes.  
**Measured behavior:** +10.76% latency drift.  
**Impact:** Peak/overall GPU latency did not fully describe sustained behavior.  
**Status:** Retained as sustained-runtime evidence.

---

## F-018 — Optical-Flow Latency Explosion

**Layer:** Classical CV baseline  
**Observation:** Dense optical flow was computationally expensive on the constrained benchmark sequence.  
**Measured behavior:** 351.59 ms mean, 533.06 ms p95, 2.84 FPS.  
**Impact:** The method was unsuitable for the tested real-time target under this configuration.  
**Status:** Measured baseline failure.

---

## F-019 — Morphological Processing Reduced Measured IoU

**Layer:** Classical CV baseline  
**Observation:** Adding morphology increased latency while reducing the measured overlap metric.  
**Measured behavior:**  
Background subtraction: 8.92 ms, IoU 0.6512.  
Background subtraction + morphology: 10.27 ms, IoU 0.6407.  
**Impact:** Additional processing added cost without improving the measured result on this sequence.  
**Status:** Measured tradeoff.

---

## F-020 — Template Matching Generalization Boundary

**Layer:** Classical CV baseline  
**Observation:** Template matching achieved perfect metrics on the deterministic synthetic sequence.  
**Measured behavior:** IoU 1.0, precision 1.0, recall 1.0, 67.29 FPS.  
**Limitation:** The experiment explicitly defines the scene as constrained and synthetic; this result does not establish general object-detection capability.  
**Impact:** Excellent constrained benchmark performance can fail to represent general deployment behavior.  
**Status:** Retained as a documented boundary condition.

---

## Evidence Policy

Where browser failures were observed during interactive validation but no frame file was saved, the record deliberately contains no invented timestamp or screenshot.

Quantitative benchmark failures are supported directly by the recorded JSON benchmark artifacts.

The purpose of this gallery is to preserve what failed, what changed, and what tradeoff was measured.
