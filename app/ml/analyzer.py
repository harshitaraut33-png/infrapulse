"""
InfraPulse — photo analysis.

Everything here runs locally, inside our own application, using the model WE fine-tuned.
No external AI/ML inference service is contacted at any point.

Three things happen for every submitted photograph:

  1. DETECT + CLASSIFY  -> which of the 4 defects is visible, and its category
  2. EXTENT             -> how much of the photo the defect covers, via Grad-CAM on our
                           own network (no second model, no extra labels)
  3. SEVERITY           -> how bad the defect looks inside that region
                           (local contrast, edge density, intensity deviation)

Those feed the documented priority formula in priority.py.
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# --------------------------------------------------------------------------------------
# Defect -> category routing. Fixed by the problem statement.
# --------------------------------------------------------------------------------------
CATEGORY_OF = {
    "spalling": "Structural",
    "stagnant_water": "Functional",
    "cracked_tiles": "Performance",
    "paint_peeling": "Performance",
}

DISPLAY_NAME = {
    "spalling": "Spalling",
    "stagnant_water": "Stagnant Water",
    "cracked_tiles": "Cracked Tiles",
    "paint_peeling": "Paint Peeling",
}

CATEGORIES = ["Structural", "Functional", "Performance"]

MODEL_PATH = os.environ.get("INFRAPULSE_MODEL", "model/infrapulse_model.pt")

_lock = threading.Lock()
_state: dict = {"model": None, "classes": None, "img_size": 224, "mean": None, "std": None}


class ModelNotLoaded(RuntimeError):
    pass


def load_model(path: str | None = None):
    """Load our fine-tuned checkpoint once and keep it in memory."""
    path = path or MODEL_PATH
    with _lock:
        if _state["model"] is not None:
            return _state["model"]

        if not os.path.exists(path):
            raise ModelNotLoaded(
                f"Model file not found at '{path}'. Train it with InfraPulse_Train.ipynb, "
                f"then save infrapulse_model.pt into the model/ folder."
            )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        classes = list(ckpt["classes"])

        net = models.resnet18(weights=None)  # architecture only; our own weights follow
        net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        _state.update(
            model=net,
            classes=classes,
            img_size=int(ckpt.get("img_size", 224)),
            mean=list(ckpt.get("norm_mean", [0.485, 0.456, 0.406])),
            std=list(ckpt.get("norm_std", [0.229, 0.224, 0.225])),
            test_accuracy=ckpt.get("test_accuracy"),
            macro_f1=ckpt.get("macro_f1"),
        )
        return net


def model_info() -> dict:
    """Summary for the /health page, so we can prove the real model is live."""
    if _state["model"] is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "classes": _state["classes"],
        "img_size": _state["img_size"],
        "test_accuracy": _state.get("test_accuracy"),
        "macro_f1": _state.get("macro_f1"),
    }


# --------------------------------------------------------------------------------------
# Grad-CAM — which pixels drove the decision
# --------------------------------------------------------------------------------------
def _grad_cam(net, tensor: torch.Tensor, class_idx: int, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Hook the last convolutional block (layer4), take the gradient of the predicted class
    score with respect to those feature maps, average each channel's gradient to get its
    importance, then sum the channels with those weights. Upsample to image size and
    normalise to 0..1.
    """
    feats: dict = {}
    grads: dict = {}

    h1 = net.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o))
    h2 = net.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    try:
        net.zero_grad(set_to_none=True)
        logits = net(tensor)
        logits[0, class_idx].backward()
        acts = feats["v"].detach()[0]            # (C, h, w)
        grad = grads["v"].detach()[0]            # (C, h, w)
    finally:
        h1.remove()
        h2.remove()

    weights = grad.mean(dim=(1, 2))              # channel importance
    cam = torch.relu((weights[:, None, None] * acts).sum(0)).numpy()

    if cam.max() <= 0:                           # degenerate case: no positive evidence
        return np.zeros(out_hw, dtype=np.float32)

    cam = cv2.resize(cam, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)


# --------------------------------------------------------------------------------------
# Severity — how bad the defect looks inside the located region
# --------------------------------------------------------------------------------------
def _severity(bgr: np.ndarray, mask: np.ndarray) -> dict:
    """
    Three visible signals, all measured inside the defect region only:

      contrast   - variation of brightness inside the region. A deep spall with exposed
                   material varies far more than a flat, lightly-stained wall.
      edges      - density of edges. Fragmentation, multiple crack lines and flaking
                   edges all raise this; a single clean mark does not.
      deviation  - how different the region is from the surrounding surface. A defect
                   that stands out sharply from its wall reads as more advanced.

    Each is scaled to 0..1 with a fixed divisor, then combined with fixed weights, so the
    same photograph always yields the same number.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    region = mask > 0.5
    outside = ~region

    if region.sum() < 64:                        # region too small to measure reliably
        return {"severity": 0.0, "contrast": 0.0, "edges": 0.0, "deviation": 0.0}

    contrast = float(gray[region].std()) / 64.0

    edges = cv2.Canny(gray, 80, 180) > 0
    edge_density = float((edges & region).sum()) / float(region.sum())
    edge_density /= 0.20                          # ~20% edge pixels is already severe

    if outside.sum() > 64:
        deviation = abs(float(gray[region].mean()) - float(gray[outside].mean())) / 64.0
    else:
        deviation = 0.0

    contrast = min(contrast, 1.0)
    edge_density = min(edge_density, 1.0)
    deviation = min(deviation, 1.0)

    severity = 0.45 * contrast + 0.35 * edge_density + 0.20 * deviation
    return {
        "severity": round(min(max(severity, 0.0), 1.0), 4),
        "contrast": round(contrast, 4),
        "edges": round(edge_density, 4),
        "deviation": round(deviation, 4),
    }


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def analyze(image_path: str) -> dict:
    """
    Analyse one photograph and return everything the rest of the system needs.

    Returns keys: defect (machine name), defect_name (display), category, confidence,
    probabilities, extent, severity, and the severity components.
    """
    net = load_model()
    classes = _state["classes"]
    size = _state["img_size"]

    pil = Image.open(image_path).convert("RGB")

    tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(_state["mean"], _state["std"]),
    ])
    tensor = tf(pil).unsqueeze(0)
    tensor.requires_grad_(True)

    # ---- classify -------------------------------------------------------------------
    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1)[0]
    idx = int(probs.argmax())
    defect = classes[idx]
    confidence = float(probs[idx])

    # ---- locate the defect and measure how much of the frame it covers ---------------
    cam = _grad_cam(net, tensor, idx, (size, size))
    extent = float((cam > 0.5).mean())

    # ---- measure how bad it looks in that region ------------------------------------
    rgb = np.array(pil.resize((size, size)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sev = _severity(bgr, cam)

    return {
        "defect": defect,
        "defect_name": DISPLAY_NAME.get(defect, defect.replace("_", " ").title()),
        "category": CATEGORY_OF.get(defect, "Performance"),
        "confidence": round(confidence, 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "extent": round(extent, 4),
        **sev,
    }
