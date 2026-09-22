# M3 Failure Inventory

These are observed failures from M3 browser/runtime validation.
No timestamps are invented where the recording evidence does not provide them.

| ID | Failure | Evidence | Status |
|---|---|---|---|
| F-001 | Embedded-NMS WebGPU inference was extremely slow | ~547.7 ms inference, no useful FPS | Observed |
| F-002 | NMS-free WebGPU model remained slow | ~462–503 ms inference, ~1.46–1.67 FPS | Observed |
| F-003 | Detached ArrayBuffer / Worker runtime error | Browser inference crashed with detached-buffer error | Observed |
| F-004 | COCO `mouse` was missed | Mouse object was present but not reliably detected | Observed |
| F-005 | Diary was classified incorrectly | Diary was mapped to person/other class | Observed |
| F-006 | Pen/highlighter was classified incorrectly | Wrong object class reported | Observed |
| F-007 | Pen/highlighter was missed | No detection in some views | Observed |
| F-008 | Microphone was not detected | Expected object produced no reliable detection | Observed |
| F-009 | Hand/object false positives | Non-target visual regions produced detections | Observed |
| F-010 | Cell phone detection was intermittent | Correct in some views, unreliable in others | Observed |
| F-011 | Laptop classification was inconsistent | Laptop/object confusion observed | Observed |

## Evidence Rule

Each failure record must eventually contain:
- What was expected
- What actually happened
- Evidence image/frame
- Runtime/model configuration
- Measured impact where applicable
- Likely failure layer
- What got worse / tradeoff
- Whether the issue is fixed, accepted, or intentionally retained

## Important Scope Note

M3 is a deployment and benchmarking module. The browser demo is evidence that the model/runtime can execute locally in the browser; it is not an arbitrary-object recognition requirement. These semantic failures therefore belong in the failure gallery rather than automatically triggering another model-training loop.
