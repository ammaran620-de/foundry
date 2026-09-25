# FOUNDRY M3 - RunsOnAnything

> **A model that runs is not automatically a model that ships.**

This browser deployment is the final portability layer of M3.

The same **YOLO26n** workload is moved from Python-based inference into a local browser environment using ONNX Runtime Web.

---

## Deployment Path

```mermaid
flowchart LR

    A["YOLO26n<br/>PyTorch CPU / FP32"]
    B["ONNX Runtime<br/>CPU"]
    C["OpenVINO<br/>FP16 CPU"]
    D["OpenVINO<br/>INT8 CPU"]
    E["OpenVINO<br/>INT8 Intel iGPU"]
    F["Browser<br/>WebGPU"]
    G["WASM<br/>Fallback"]

    A --> B --> C --> D --> E --> F
    F -. fallback .-> G