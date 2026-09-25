"""Self-check for the model export path. Run where torch + onnxruntime exist:
    python test_model_export.py

Asserts the invariant both export fixes rely on:
  models/play_model.onnx must be the same function as models/play_model_best.pt
  (train.py exports best weights, and evaluate.py's .pt -> .onnx conversion round-trips).
"""
import os
import tempfile

import numpy as np
import torch
import onnxruntime as ort

from features import FEATURE_DIM
from nn import load_play_model, export_onnx

x = np.random.RandomState(0).randn(4, FEATURE_DIM).astype(np.float32)

model = load_play_model('models/play_model_best.pt')
with torch.no_grad():
    p_ref, v_ref = model(torch.from_numpy(x))
p_ref, v_ref = p_ref.numpy(), v_ref.numpy()


def onnx_out(path):
    s = ort.InferenceSession(path)
    out = s.run(None, {s.get_inputs()[0].name: x})
    return out[0], out[1]


# Bug B: the exported .onnx must match the best checkpoint, not final-epoch weights
p, v = onnx_out('models/play_model.onnx')
assert np.allclose(p, p_ref, atol=1e-4) and np.allclose(v, v_ref, atol=1e-4), \
    "play_model.onnx does not match play_model_best.pt (exported from final weights?)"

# Bug A: the .pt -> .onnx conversion evaluate.py does before loading must round-trip
with tempfile.TemporaryDirectory() as d:
    tmp = os.path.join(d, 'candidate.onnx')
    export_onnx(model, tmp, FEATURE_DIM)
    p, v = onnx_out(tmp)
assert np.allclose(p, p_ref, atol=1e-4) and np.allclose(v, v_ref, atol=1e-4), \
    ".pt -> .onnx conversion changed the model"

print("OK: play_model.onnx matches play_model_best.pt and .pt conversion round-trips")
