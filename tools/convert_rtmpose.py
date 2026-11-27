"""Export MMPose's RTMPose-m trained on Human-Art + COCO to ONNX for the browser.

Usage (with mmpose 1.3, mmcv-lite 2.2, torch, munkres, xtcocotools installed):
  python -I tools/convert_rtmpose.py out.onnx
Input: 1×3×256×192 RGB, normalised with mean (123.675, 116.28, 103.53) / std (58.395, 57.12, 57.375).
Outputs: simcc_x 1×17×384, simcc_y 1×17×512 (SimCC, split ratio 2).
"""
import os
import sys
import types

# mmpose imports mmdet and mmcv's compiled ops for heads RTMPose doesn't use; stubs avoid building them.
sys.modules["mmdet"] = types.ModuleType("mmdet")
sys.modules["mmdet.utils"] = types.SimpleNamespace(ConfigType=dict, reduce_mean=lambda x: x)
sys.modules["mmcv.ops"] = types.SimpleNamespace(MultiScaleDeformableAttention=object, DeformConv2d=object, ModulatedDeformConv2d=object)

import mmpose
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpose.registry import MODELS
from mmpose.utils import register_all_modules

CONFIG = "configs/body_2d_keypoint/rtmpose/humanart/rtmpose-m_8xb256-420e_humanart-256x192.py"
CKPT = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth"

register_all_modules()
cfg = Config.fromfile(os.path.join(os.path.dirname(mmpose.__file__), ".mim", CONFIG))
model = MODELS.build(cfg.model).eval()
load_checkpoint(model, CKPT, map_location="cpu")


class Export(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.head(self.m.backbone(x))


torch.onnx.export(Export(model), torch.zeros(1, 3, 256, 192), sys.argv[1], input_names=["input"],
                  output_names=["simcc_x", "simcc_y"], opset_version=17)
print("wrote", sys.argv[1])
