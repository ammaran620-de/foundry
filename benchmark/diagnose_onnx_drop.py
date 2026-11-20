"""
Step 1 of the ONNX accuracy-drop hunt. Reads, does not evaluate.
Run from the foundry repo root:  python diagnose_onnx_drop.py
Paste the full output back.
"""
from pathlib import Path
import json

import onnx

ROOT = Path.cwd()
ONNX_PATH = ROOT / "models" / "yolo26n.onnx"
OV_XML = ROOT / "models" / "yolo26n_openvino_model" / "yolo26n.xml"

m = onnx.load(str(ONNX_PATH))

print("== export metadata (Ultralytics stores export args here) ==")
for p in m.metadata_props:
    print(f"{p.key}: {p.value[:300]}")

print("\n== graph inputs / outputs ==")
for t in list(m.graph.input) + list(m.graph.output):
    dims = [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
    print(t.name, dims)

ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("\n== op counts (look for NonMaxSuppression, TopK) ==")
for k in ("NonMaxSuppression", "TopK", "Greater", "GatherElements"):
    print(f"{k}: {ops.get(k, 0)}")

print("\n== NMS / threshold constants ==")
for n in m.graph.node:
    if n.op_type == "NonMaxSuppression":
        print("NMS node inputs:", list(n.input))
for init in m.graph.initializer:
    if init.dims in ([], [1]) and init.data_type == 1:  # scalar float
        vals = onnx.numpy_helper.to_array(init)
        print(f"scalar const {init.name}: {vals}")

print("\n== opset ==")
print([ (o.domain or 'ai.onnx', o.version) for o in m.opset_import ])

if OV_XML.exists():
    txt = OV_XML.read_text(errors="ignore")
    print("\n== OpenVINO xml: NMS-like layers ==")
    for key in ("NonMaxSuppression", "TopK"):
        print(key, txt.count(f'type="{key}"'))
