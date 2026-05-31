"""
camera.py — Camera driver abstraction
======================================
Defines a common CameraDriver interface and two concrete implementations:

  • RealSenseDriver  — Intel RealSense L515 (RGB + aligned depth via pyrealsense2)
  • USBCameraDriver  — Any USB / built-in webcam via OpenCV (depth not available)

Usage
-----
    from camera import make_driver

    cam = make_driver("realsense")   # or "usb" / "usb:1"
    try:
        while True:
            color, depth = cam.read_frame()
            if color is None:
                continue
            # color  → np.ndarray (H, W, 3) BGR
            # depth  → rs.depth_frame aligned to color  (None for USB)
    finally:
        cam.release()
"""

from __future__ import annotations

import abc
from typing import Tuple, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ColorFrame = np.ndarray          # (H, W, 3) uint8 BGR
DepthFrame  = Optional[object]   # rs.depth_frame | None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CameraDriver(abc.ABC):
    """Common interface for all camera back-ends."""

    @abc.abstractmethod
    def read_frame(self) -> Tuple[Optional[ColorFrame], DepthFrame]:
        """
        Return (color_bgr, depth_frame).

        color_bgr   : numpy array (H, W, 3) or None on failure.
        depth_frame : back-end-specific depth object, or None when unavailable.
        """

    @abc.abstractmethod
    def release(self) -> None:
        """Free all hardware/OS resources."""

    # Convenience: allow use as a context manager
    def __enter__(self) -> "CameraDriver":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # Legacy shim so existing code that calls  cap.read()  keeps working
    def read(self) -> Tuple[bool, Optional[ColorFrame]]:
        color, _ = self.read_frame()
        return (color is not None), color


# ---------------------------------------------------------------------------
# RealSense back-end
# ---------------------------------------------------------------------------

class RealSenseDriver(CameraDriver):
    """
    Intel RealSense L515 (or any RealSense with RGB + depth streams).

    Activates:
      • Color stream  – 1280×720 @ 30 fps (BGR8)
      • Depth stream  – 640×480  @ 30 fps (Z16, millimetres)

    depth_frame returned by read_frame() is an rs.depth_frame already
    aligned to the colour frame, so depth_frame[y][x] corresponds to
    the same pixel as color_bgr[y][x].

    Depth helper
    ------------
    Use get_face_depth(depth_frame, x, y, w, h) (module-level function)
    to obtain (median_metres, std_metres) for a bounding-box ROI.
    """

    #: RealSense depth scale for the L515 (metres per raw unit)
    DEPTH_SCALE: float = 0.00025

    def __init__(
        self,
        color_width: int  = 1280,
        color_height: int = 720,
        color_fps: int    = 30,
        depth_width: int  = 640,
        depth_height: int = 480,
        depth_fps: int    = 30,
        timeout_ms: int   = 5000,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed.  "
                "Install it with:  pip install pyrealsense2"
            ) from exc

        self._rs = rs
        self._timeout_ms = timeout_ms

        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
        cfg.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16,  depth_fps)

        self._pipeline = rs.pipeline()
        self._pipeline.start(cfg)

        # Align depth to colour so they share the same pixel grid
        self._align = rs.align(rs.stream.color)

    # ------------------------------------------------------------------

    def read_frame(self) -> Tuple[Optional[ColorFrame], DepthFrame]:
        try:
            frames  = self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
            aligned = self._align.process(frames)
        except Exception:
            return None, None

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame:
            return None, None

        color_bgr = np.asanyarray(color_frame.get_data())
        return color_bgr, depth_frame

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# USB / built-in webcam back-end
# ---------------------------------------------------------------------------

class USBCameraDriver(CameraDriver):
    """
    Generic webcam via cv2.VideoCapture.

    depth_frame is always None — use software-based anti-spoofing or
    skip depth checks when running without a RealSense device.

    Parameters
    ----------
    index : int | str
        cv2.VideoCapture source.  0 = default cam, 1 = second cam,
        or a video file path / RTSP URL.
    width, height : optional resolution hints (best-effort).
    """

    def __init__(
        self,
        index: int | str = 0,
        width: Optional[int]  = None,
        height: Optional[int] = None,
    ) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index/path: {index!r}")

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self) -> Tuple[Optional[ColorFrame], DepthFrame]:
        ok, frame = self._cap.read()
        return (frame if ok else None), None

    def release(self) -> None:
        self._cap.release()


# ---------------------------------------------------------------------------
# Depth utility (used only with RealSenseDriver)
# ---------------------------------------------------------------------------

def get_face_depth(
    depth_frame,
    x: int, y: int, w: int, h: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (median_metres, std_metres) for a face bounding-box ROI.

    Uses the median to ignore pixels with no LiDAR return (raw value 0).
    Returns (None, None) when no valid depth pixels exist in the ROI.

    Parameters
    ----------
    depth_frame : aligned rs.depth_frame from RealSenseDriver.read_frame()
    x, y, w, h  : bounding box in colour-frame pixel coordinates
    """
    import numpy as np

    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_image *= RealSenseDriver.DEPTH_SCALE   # raw units → metres

    roi   = depth_image[y : y + h, x : x + w]
    valid = roi[roi > 0]

    if len(valid) == 0:
        return None, None

    return float(np.median(valid)), float(np.std(valid))


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_driver(backend: str = "realsense", **kwargs) -> CameraDriver:
    """
    Instantiate a camera driver by name.

    backend
    -------
    "realsense"          → RealSenseDriver(**kwargs)
    "usb"  or "usb:N"   → USBCameraDriver(index=N, **kwargs)   (N defaults to 0)

    Examples
    --------
        cam = make_driver("realsense")
        cam = make_driver("usb")          # default webcam
        cam = make_driver("usb:1")        # second webcam
        cam = make_driver("usb", index=2, width=1280, height=720)
    """
    backend = backend.strip().lower()

    if backend == "realsense":
        return RealSenseDriver(**kwargs)

    if backend.startswith("usb"):
        parts = backend.split(":", 1)
        index = int(parts[1]) if len(parts) == 2 else kwargs.pop("index", 0)
        return USBCameraDriver(index=index, **kwargs)

    raise ValueError(
        f"Unknown backend {backend!r}.  "
        "Choose 'realsense', 'usb', or 'usb:<index>'."
    )
