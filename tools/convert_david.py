# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["onnx>=1.17", "onnxruntime>=1.20", "sympy", "packaging"]
# ///
"""Convert Microsoft DAViD's surface-normal model (ViT-B, 512×512, MIT) to fp16 for the browser.

Usage: uv run tools/convert_david.py <normal-model-vitb16_384.onnx> <out.onnx>
Source: https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/normal-model-vitb16_384.onnx
Inputs/outputs stay fp32 (keep_io_types), so the page feeds float32 as before.
"""
import sys

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

src, dst = sys.argv[1], sys.argv[2]
onnx.save(convert_float_to_float16(onnx.load(src), keep_io_types=True), dst)
print("wrote", dst)
