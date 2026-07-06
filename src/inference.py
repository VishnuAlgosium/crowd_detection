"""
inference.py — Faster R-CNN inference worker thread
Runs GPU inference continuously on the latest frame,
publishes results without blocking the display loop.
"""

import threading
import time
import torch
import torch.backends.cudnn as cudnn
from torchvision.ops import nms
import cv2

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


def load_model(cfg: dict):
    device = torch.device(cfg["model"]["device"])

    if device.type == "cuda":
        cudnn.benchmark     = True
        cudnn.deterministic = False

    print("[Model] Loading Faster R-CNN ResNet50...")
    weights   = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model     = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    transform = weights.transforms()
    print(f"[Model] Loaded on: {device}")
    return model, transform, device


class InferWorker:
    def __init__(self, model, transform, device, capture, cfg: dict):
        self.model     = model
        self.transform = transform
        self.device    = device
        self.capture   = capture

        self.confidence = cfg["model"]["confidence_threshold"]
        self.person_id  = cfg["model"]["person_class_id"]
        self.iou_threshold = cfg["model"]["iou_threshold"]

        self._lock       = threading.Lock()
        self._detections = []
        self._running    = True

        threading.Thread(target=self._worker, daemon=True).start()
        print("[Infer] Worker started")

    def _worker(self):
        while self._running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.to(self.device, non_blocking=True)
            tensor = self.transform(tensor)

            with torch.no_grad():
                output = self.model([tensor])[0]

            boxes  = output["boxes"]    # keep as tensor — NMS needs tensor
            scores = output["scores"]
            labels = output["labels"]

            # Step 1 — filter persons above confidence (still tensors)
            mask   = (labels == self.person_id) & (scores >= self.confidence)
            boxes  = boxes[mask]
            scores = scores[mask]

            # Step 2 — NMS on tensors (convert to numpy AFTER this)
            if len(boxes) > 0:
                keep   = nms(boxes, scores, iou_threshold=self.iou_threshold)
                boxes  = boxes[keep]
                scores = scores[keep]

            # Step 3 — now safe to convert to numpy
            results = list(zip(
                boxes.float().cpu().numpy(),
                scores.float().cpu().tolist(),
            ))

            with self._lock:
                self._detections = results

    def get_detections(self):
        with self._lock:
            return list(self._detections)

    def stop(self):
        self._running = False
        print("[Infer] Worker stopped")
