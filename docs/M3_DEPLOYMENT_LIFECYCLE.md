# M3 — RunsOnAnything
## Deployment & Optimization Lifecycle

> One computer-vision model. Multiple runtimes, hardware targets, and deployment environments.

```mermaid
flowchart LR

    A["FOUNDRY M3<br/><b>RUNSONANYTHING</b><br/><br/>One computer-vision model.<br/>Multiple runtimes, hardware targets,<br/>and deployment environments."]

    B["<b>01 — PROBLEM</b><br/><br/><b>A model that runs<br/>is not automatically<br/>a model that ships.</b><br/><br/>Accuracy alone is not enough:<br/>speed • memory • stability • hardware fit"]

    C["<b>02 — BASELINE</b><br/><br/><b>YOLO26n</b><br/>PyTorch CPU / FP32<br/><br/>Establish a reproducible<br/>reference point."]

    D["<b>03 — OPTIMIZE</b><br/><br/><b>Portable + efficient execution</b><br/><br/>ONNX Runtime<br/>↓<br/>OpenVINO FP16<br/>↓<br/><b>OpenVINO INT8</b><br/>↓<br/><b>Intel iGPU</b>"]

    E["<b>04 — MEASURE</b><br/><br/><b>Performance</b><br/>Mean • p50 • p95 • FPS<br/><br/><b>Quality</b><br/>mAP • Precision • Recall<br/><br/><b>System</b><br/>RAM • threads • sustained behavior"]

    F["<b>05 — FIND THE LIMIT</b><br/><br/><b>CPU thread scaling</b><br/><br/>4 threads → <b>19.06 FPS</b><br/>6 threads → 16.09 FPS<br/>10 threads → 17.09 FPS<br/><br/><b>More threads ≠ more FPS</b>"]

    G["<b>06 — VALIDATE + BREAK</b><br/><br/><b>10-minute sustained testing</b><br/>CPU → 19.87 FPS<br/>Intel iGPU → 61.78 FPS<br/><br/>0 timeouts on both runs<br/><br/>Failures are recorded,<br/>not hidden."]

    H["<b>07 — DEPLOY</b><br/><br/><b>Browser / WebGPU</b><br/>Local ONNX inference<br/>WebAssembly fallback<br/>Webcam + still image<br/>~9.4 MB model<br/><br/><b>No frame-upload API</b>"]

    I["<b>FINAL ENGINEERING RESULT</b><br/><br/><b>Accuracy + Latency + FPS<br/>Memory + Hardware + Reliability</b><br/><br/>The goal is not simply<br/>“make the model faster.”<br/><br/>The goal is to find a deployment<br/>point that actually makes sense."]

    A --> B --> C --> D --> E --> F --> G --> H --> I

    classDef hero fill:#073B3A,stroke:#0F766E,color:#FFFFFF,stroke-width:3px;
    classDef problem fill:#FFF7ED,stroke:#C2410C,color:#7C2D12,stroke-width:3px;
    classDef stage fill:#F0FDFA,stroke:#0F766E,color:#073B3A,stroke-width:2px;
    classDef finding fill:#ECFDF5,stroke:#15803D,color:#14532D,stroke-width:3px;
    classDef outcome fill:#102A2A,stroke:#0F766E,color:#FFFFFF,stroke-width:3px;

    class A hero;
    class B problem;
    class C,D,E,H stage;
    class F,G finding;
    class I outcome;