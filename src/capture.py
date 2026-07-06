"""
capture.py — FFmpeg NVDEC RTSP capture thread
Decodes frames on GPU, always holds the latest frame.
"""

import subprocess
import threading
import time
import numpy as np


class RTSPCapture:
    def __init__(self, cfg: dict):
        cam    = cfg["camera"]
        ffmpeg = cfg["ffmpeg"]

        self.width      = cam["width"]
        self.height     = cam["height"]
        self.fps        = cam["fps"]
        self.frame_size = self.width * self.height * 3

        cmd = ["ffmpeg"]

        # GPU decode
        if ffmpeg.get("hwaccel") == "cuda":
            cmd += ["-hwaccel", "cuda", "-c:v", ffmpeg.get("codec", "hevc_cuvid")]

        # Low-latency RTSP
        cmd += ["-rtsp_transport", cam.get("rtsp_transport", "tcp")]
        if ffmpeg.get("low_delay", True):
            cmd += ["-fflags", "nobuffer", "-flags", "low_delay"]

        cmd += [
            "-probesize",       str(ffmpeg.get("probe_size", 32)),
            "-analyzeduration", str(ffmpeg.get("analyze_duration", 0)),
            "-i",               cam["rtsp_url"],
            "-pix_fmt",         "bgr24",
            "-f",               "rawvideo",
            "-",
        ]

        self._pipe = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10 ** 8,
        )

        self._lock    = threading.Lock()
        self._frame   = None
        self._ok      = False
        self._running = True

        threading.Thread(target=self._reader, daemon=True).start()
        print(f"[Capture] {self.width}x{self.height} @ {self.fps} FPS | "
              f"hwaccel={ffmpeg.get('hwaccel')} codec={ffmpeg.get('codec')}")

    def _reader(self):
        while self._running:
            raw = self._pipe.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                time.sleep(0.005)
                continue
            frame = np.frombuffer(raw, np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self._lock:
                self._frame = frame
                self._ok    = True

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def get_props(self):
        return self.width, self.height, self.fps

    def release(self):
        self._running = False
        if self._pipe:
            self._pipe.kill()
        print("[Capture] Released")
