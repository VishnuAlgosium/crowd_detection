"""
People Detection + Heatmap using Faster R-CNN ResNet50

Architecture:
  Thread 1 (RTSPCapture)  — grabs frames continuously, always holds latest
  Thread 2 (InferWorker)  — runs GPU inference continuously on latest frame
  Main thread             — reads results, draws, displays at full speed

This means capture and inference never block each other.

Run:
    python people_detect_fast.py

Requirements:
    pip install torch torchvision opencv-python
"""

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import threading
import time
from collections import deque
import subprocess


from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.7
PERSON_CLASS_ID      = 1

HEATMAP_DECAY        = 0.94
HEATMAP_ALPHA        = 0.5
HEATMAP_SIGMA        = 40

RTSP_URL = "rtsp://admin:diffuse123@192.168.0.183:554/cam/realmonitor?channel=1&subtype=0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    cudnn.benchmark     = True
    cudnn.deterministic = False


# ─────────────────────────────────────────
# THREAD 1 — RTSP CAPTURE
# Always holds the latest frame
# ─────────────────────────────────────────
class RTSPCapture:
    def __init__(self, url):
        self.url = url

        # Get stream info first
        self.width  = 1280
        self.height = 720
        self.fps    = 15
        print(f"[INFO] Stream resolution: {self.width}x{self.height}")

        # FFmpeg GPU decode command
        self.command = [
            "ffmpeg",

            # GPU decode
            "-hwaccel", "cuda",
            "-c:v", "hevc_cuvid",

            # Low latency RTSP
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",

            "-probesize", "32",
            "-analyzeduration", "0",

            "-i", self.url,

            # Output raw frames
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-"
        ]

        self.pipe = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        self.frame_size = 1280 * 720 * 3

        self.lock    = threading.Lock()
        self.frame   = None
        self.ok      = False
        self.running = True

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            raw = self.pipe.stdout.read(self.frame_size)

            if len(raw) != self.frame_size:
                time.sleep(0.005)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape(
                (self.height, self.width, 3)
            )

            with self.lock:
                self.frame = frame
                self.ok    = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None

            return self.ok, self.frame.copy()

    def get_props(self):
        return self.width, self.height, self.fps

    def release(self):
        self.running = False

        if self.pipe:
            self.pipe.kill()


# ─────────────────────────────────────────
# THREAD 2 — INFERENCE WORKER
# Pulls latest frame, runs GPU inference,
# publishes results — never blocks display
# ─────────────────────────────────────────

class InferWorker:
    def __init__(self, model, transform, capture):
        self.model     = model
        self.transform = transform
        self.capture   = capture

        self.lock       = threading.Lock()
        self.detections = []   # latest results available to main thread
        self.running    = True

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            tensor = tensor.float().to(DEVICE, non_blocking=True)
            tensor = tensor / 255.0

            tensor = self.transform(tensor)

            with torch.no_grad():
                output = self.model([tensor])[0]

            boxes  = output["boxes"].float().cpu().numpy()
            scores = output["scores"].float().cpu().numpy()
            labels = output["labels"].cpu().numpy()

            results = [
                (box, score)
                for box, score, label in zip(boxes, scores, labels)
                if label == PERSON_CLASS_ID and score >= CONFIDENCE_THRESHOLD
            ]

            with self.lock:
                self.detections = results

    def get_detections(self):
        with self.lock:
            return list(self.detections)

    def stop(self):
        self.running = False


# ─────────────────────────────────────────
# HEATMAP HELPERS
# ─────────────────────────────────────────

def make_gaussian_kernel(sigma, size=None):
    if size is None:
        size = int(6 * sigma + 1) | 1
    ax     = np.arange(-(size // 2), size // 2 + 1)
    gauss  = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss, gauss)
    return (kernel / kernel.max()).astype(np.float32)


def add_heat(heatmap, cx, cy, sigma=HEATMAP_SIGMA, value=1.0):
    h, w   = heatmap.shape
    kernel = make_gaussian_kernel(sigma)
    ks     = kernel.shape[0]
    half   = ks // 2

    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    kx0 = max(0, -x0);  kx1 = ks - max(0, x1 - w)
    ky0 = max(0, -y0);  ky1 = ks - max(0, y1 - h)
    x0  = max(0, x0);   x1  = min(w, x1)
    y0  = max(0, y0);   y1  = min(h, y1)

    if x1 > x0 and y1 > y0:
        heatmap[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1] * value


def render_heatmap(frame, heatmap):
    norm = np.clip(heatmap, 0, None)
    if norm.max() > 0:
        norm = norm / norm.max()
    heat_u8  = (norm * 255).astype(np.uint8)
    heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    mask     = (heat_u8 > 10).astype(np.float32)[:, :, None]
    blended  = (frame * (1 - HEATMAP_ALPHA * mask) +
                heat_col * HEATMAP_ALPHA * mask).astype(np.uint8)
    return blended


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

print("[INFO] Loading Faster R-CNN ResNet50 model...")
weights   = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model     = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
model.to(DEVICE)
transform = weights.transforms()
print(f"[INFO] Model loaded on: {DEVICE}")


# ─────────────────────────────────────────
# START THREADS
# ─────────────────────────────────────────

print("[INFO] Connecting to RTSP stream...")
cap = RTSPCapture(RTSP_URL)
time.sleep(1.5)

cam_w, cam_h, cam_fps = cap.get_props()
print(f"[INFO] Camera: {cam_w}x{cam_h} @ {cam_fps:.0f} FPS")

worker  = InferWorker(model, transform, cap)
heatmap = np.zeros((cam_h, cam_w), dtype=np.float32)

fps_times = deque(maxlen=30)

print("[INFO] Starting — press Q to quit")


# ─────────────────────────────────────────
# MAIN LOOP  (display only — never blocks on inference)
# ─────────────────────────────────────────

while True:
    t0 = time.perf_counter()

    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.005)
        continue

    # Get latest detections from inference thread (non-blocking)
    detections = worker.get_detections()

    # ── HEATMAP DECAY ───────────────────────────────────────────
    heatmap *= HEATMAP_DECAY

    # ── DRAW + UPDATE HEATMAP ───────────────────────────────────
    for (x1, y1, x2, y2), score in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        add_heat(heatmap, (x1 + x2) // 2, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)

    # ── BLEND HEATMAP ───────────────────────────────────────────
    frame = render_heatmap(frame, heatmap)

    # ── FPS ─────────────────────────────────────────────────────
    t1 = time.perf_counter()
    fps_times.append(t1 - t0)
    fps = 1.0 / (sum(fps_times) / len(fps_times))

    # ── HUD ─────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS:    {fps:.1f}",         (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"People: {len(detections)}", (20,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
    cv2.putText(frame, f"Res:    {cam_w}x{cam_h}",  (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("People Detection + Heatmap", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────

worker.stop()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program stopped")