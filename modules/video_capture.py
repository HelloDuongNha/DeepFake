import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable
import platform
import threading

# Only import Windows-specific library if on Windows
if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class VideoCapturer:
    def __init__(self, device_index: int):
        if isinstance(device_index, bool):
            raise TypeError("Camera device index must be an integer, not bool")
        try:
            self.device_index = int(device_index)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                f"Camera device index must be an integer, got {device_index!r}"
            ) from error
        self.frame_callback = None
        self._current_frame = None
        self._prefetched_frame = None
        self._frame_ready = threading.Event()
        self._empty_read_count = 0
        self.is_running = False
        self.cap = None
        # Actual values reported by the camera after configuration
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

        # Initialize Windows-specific components if on Windows
        if platform.system() == "Windows":
            self.graph = FilterGraph()
            # Verify device exists
            devices = self.graph.get_input_devices()
            if self.device_index >= len(devices):
                raise ValueError(
                    f"Invalid device index {device_index}. Available devices: {len(devices)}"
                )

    def start(self, width: int = 960, height: int = 540, fps: int = 60) -> bool:
        """Initialize and start video capture"""
        try:
            print(
                f"[VideoCapturer] Opening device index={self.device_index} "
                f"platform={platform.system()}",
                flush=True,
            )
            if platform.system() == "Windows":
                # device_index comes from pygrabber.FilterGraph (DirectShow
                # enumeration), so open with DSHOW first to preserve mapping.
                # MSMF and DirectShow enumerate cameras in different orders, so
                # opening MSMF with a DSHOW index silently selects the wrong
                # camera. MSMF/ANY remain as fallbacks for cameras DSHOW can't
                # open.
                #
                # Pass codec + resolution + fps as construction params (OpenCV
                # 4.6+). DSHOW locks the pixel format at open time and ignores
                # later cap.set(CAP_PROP_FOURCC, ...) — without this, DSHOW
                # falls back to uncompressed YUYV at 1080p, which is USB-
                # bandwidth-limited to ~5 fps. Setting MJPG at construction
                # negotiates compressed frames from the first read.
                mjpg = cv2.VideoWriter_fourcc(*'MJPG')
                open_params = [
                    cv2.CAP_PROP_FOURCC, mjpg,
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, fps,
                ]
                capture_methods = [
                    (self.device_index, cv2.CAP_DSHOW),
                    (self.device_index, cv2.CAP_MSMF),
                    (self.device_index, cv2.CAP_ANY),
                ]

                for dev_id, backend in capture_methods:
                    try:
                        self.cap = cv2.VideoCapture(dev_id, backend, open_params)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                    except Exception:
                        continue
            elif platform.system() == "Linux":
                self.cap = cv2.VideoCapture(f"/dev/video{self.device_index}")
            elif platform.system() == "Darwin":
                self.cap = cv2.VideoCapture(
                    int(self.device_index), cv2.CAP_AVFOUNDATION
                )
            else:
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                print(
                    f"[VideoCapturer] cap.isOpened() failed for device "
                    f"index={self.device_index}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Failed to open camera index {self.device_index}"
                )

            # Belt-and-braces: also set via cap.set() for backends that honor
            # post-open changes (MSMF, V4L2). DSHOW ignores these, but the
            # construction params above already handled it.
            if platform.system() != "Windows":
                # AVFoundation chooses the native camera pixel format. Forcing
                # MJPG can leave built-in FaceTime cameras open but returning
                # empty frames, which appears as a black preview.
                if platform.system() != "Darwin":
                    self.cap.set(
                        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')
                    )
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Read back resolution (usually reliable)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # CAP_PROP_FPS is unreliable on DirectShow — often reports 30
            # even when the camera delivers 60.  Measure empirically by
            # timing a burst of frames.
            reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.actual_fps = self._measure_fps(warmup=10, sample=30,
                                                fallback=reported_fps or fps)

            if self._prefetched_frame is None:
                # Some AVFoundation devices open asynchronously. Give the
                # built-in camera a short grace period before declaring it
                # unusable.
                for _ in range(50):
                    ret, frame = self.cap.read()
                    if ret and isinstance(frame, np.ndarray) and frame.size > 0:
                        self._prefetched_frame = frame
                        break
                    time.sleep(0.02)
                if self._prefetched_frame is None:
                    print(
                        f"[VideoCapturer] Camera index={self.device_index} opened "
                        "but returned no valid frame",
                        flush=True,
                    )
                    self.cap.release()
                    self.cap = None
                    return False

            print(f"[VideoCapturer] {self.actual_width}x{self.actual_height} "
                  f"@ {self.actual_fps:.1f}fps (reported={reported_fps:.0f})",
                  flush=True)

            self.is_running = True
            return True

        except Exception as e:
            print(f"Failed to start capture: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera"""
        if not self.is_running:
            return False, None

        if self.cap is None:
            return False, None

        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            self._current_frame = frame
            return True, frame

        ret, frame = self.cap.read()
        if ret and isinstance(frame, np.ndarray) and frame.size > 0:
            self._empty_read_count = 0
            self._current_frame = frame
            if self.frame_callback:
                self.frame_callback(frame)
            return True, frame
        self._log_empty_read()
        return False, None

    def release(self) -> None:
        """Stop capture and release resources"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _log_empty_read(self) -> None:
        self._empty_read_count += 1
        if self._empty_read_count == 1 or self._empty_read_count % 30 == 0:
            print(
                f"[VideoCapturer] Empty frame from camera index="
                f"{self.device_index} (count={self._empty_read_count})",
                flush=True,
            )

    def _measure_fps(self, warmup: int = 10, sample: int = 30,
                     fallback: float = 30.0) -> float:
        """Read warmup+sample frames and return measured FPS.

        This is more reliable than CAP_PROP_FPS which often lies on
        DirectShow.  Takes ~0.5-1s at startup but gives a ground-truth
        number for adaptive polling/detection intervals.
        """
        try:
            for _ in range(warmup):
                ret, frame = self.cap.read()
                if ret and isinstance(frame, np.ndarray) and frame.size > 0:
                    self._prefetched_frame = frame
            t0 = time.perf_counter()
            valid_frames = 0
            for _ in range(sample):
                ret, frame = self.cap.read()
                if ret and isinstance(frame, np.ndarray) and frame.size > 0:
                    valid_frames += 1
                    self._prefetched_frame = frame
            elapsed = time.perf_counter() - t0
            if elapsed <= 0 or valid_frames == 0:
                return fallback
            return valid_frames / elapsed
        except Exception:
            return fallback

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback for frame processing"""
        self.frame_callback = callback
