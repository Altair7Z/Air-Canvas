"""
hand_tracker.py
---------------
Handles real-time hand landmark detection using MediaPipe Tasks API
(mediapipe >= 1.0.0 / HandLandmarker).

Provides coordinate extraction, EMA smoothing, and gesture classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 1.0 Tasks API — correct import paths
_BaseOptions = mp.tasks.BaseOptions
_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

# Default model path — bundled alongside this file.
_DEFAULT_MODEL = Path(__file__).parent / "hand_landmarker.task"


class Gesture(Enum):
    """Recognised hand gesture states used by the state machine."""
    NONE = auto()       # No hand detected
    DRAWING = auto()    # Index finger only — draw mode
    HOVER = auto()      # Index + middle fingers — UI selection mode
    ERASER = auto()     # Open palm (4‑5 fingers up) — erase / clear mode


@dataclass
class HandLandmarks:
    """Snapshot of a single hand's relevant landmarks for one frame."""
    # Raw pixel coordinates of index fingertip (landmark 8)
    index_tip: tuple[int, int] = (0, 0)
    # Raw pixel coordinates of middle fingertip (landmark 12)
    middle_tip: tuple[int, int] = (0, 0)
    # EMA-smoothed index fingertip used for drawing
    smooth_tip: tuple[int, int] = (0, 0)
    gesture: Gesture = Gesture.NONE
    # Full normalised landmark list (NormalizedLandmark objects)
    raw_landmarks: list = field(default_factory=list)


class HandTracker:
    """
    Wraps the MediaPipe Tasks HandLandmarker to detect a single hand and expose:
      - Fingertip coordinates (raw + EMA-smoothed)
      - Gesture classification via a simple rule-based state machine

    The model file ``hand_landmarker.task`` must be present in the same
    directory as this script (or provide an explicit *model_path*).

    Parameters
    ----------
    model_path : Path or str
        Path to the ``hand_landmarker.task`` model bundle.
    max_hands : int
        Maximum number of hands to track (default 1 for performance).
    detection_confidence : float
        Minimum confidence for hand detection.
    presence_confidence : float
        Minimum confidence for hand presence in landmark detection.
    tracking_confidence : float
        Minimum confidence for landmark tracking.
    ema_alpha : float
        Exponential moving average factor (0 < α ≤ 1).
        Lower = smoother but laggier; higher = more responsive.
    """

    # MediaPipe landmark indices (consistent across legacy and tasks API)
    _INDEX_TIP   = 8
    _INDEX_PIP   = 6
    _MIDDLE_TIP  = 12
    _MIDDLE_PIP  = 10
    _RING_TIP    = 16
    _RING_PIP    = 14
    _PINKY_TIP   = 20
    _PINKY_PIP   = 18

    # Hand connection pairs (index pairs into the 21-landmark list)
    _CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17), (0, 5),
    ]

    def __init__(
        self,
        model_path: Path | str = _DEFAULT_MODEL,
        max_hands: int = 1,
        detection_confidence: float = 0.70,
        presence_confidence: float = 0.70,
        tracking_confidence: float = 0.70,
        ema_alpha: float = 0.35,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Run the following command to download it:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',"
                " 'hand_landmarker.task')\""
            )

        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = _HandLandmarker.create_from_options(options)
        self._ema_alpha = ema_alpha
        self._smooth_x: float = 0.0
        self._smooth_y: float = 0.0
        self._initialized: bool = False
        self._frame_ts_ms: int = 0  # synthetic monotonic timestamp in ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[Optional[HandLandmarks], np.ndarray]:
        """
        Process a single BGR webcam frame.

        Returns
        -------
        landmarks : HandLandmarks or None
            Detected hand data, or None if no hand found.
        annotated_frame : np.ndarray
            Frame with a simple hand skeleton drawn on it (BGR).
        """
        h, w = frame.shape[:2]

        # Advance synthetic timestamp by ~33 ms per frame
        self._frame_ts_ms += 33

        # MediaPipe Tasks requires an mp.Image object in BGR input
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result = self._detector.detect_for_video(mp_image, self._frame_ts_ms)

        annotated = frame.copy()

        if not result.hand_landmarks:
            self._initialized = False
            return None, annotated

        # Use the first detected hand
        lm = result.hand_landmarks[0]

        # Draw skeleton manually (Tasks API doesn't bundle drawing_utils)
        self._draw_landmarks(annotated, lm, w, h)

        def to_px(idx: int) -> tuple[int, int]:
            return int(lm[idx].x * w), int(lm[idx].y * h)

        index_tip  = to_px(self._INDEX_TIP)
        middle_tip = to_px(self._MIDDLE_TIP)
        smooth_tip = self._apply_ema(index_tip)
        gesture    = self._classify_gesture(lm, h)

        return HandLandmarks(
            index_tip=index_tip,
            middle_tip=middle_tip,
            smooth_tip=smooth_tip,
            gesture=gesture,
            raw_landmarks=lm,
        ), annotated

    def release(self) -> None:
        """Close the detector and free resources."""
        self._detector.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_ema(self, tip: tuple[int, int]) -> tuple[int, int]:
        """Apply exponential moving average to smooth jitter."""
        x, y = tip
        if not self._initialized:
            self._smooth_x = float(x)
            self._smooth_y = float(y)
            self._initialized = True
        else:
            α = self._ema_alpha
            self._smooth_x = α * x + (1 - α) * self._smooth_x
            self._smooth_y = α * y + (1 - α) * self._smooth_y
        return int(self._smooth_x), int(self._smooth_y)

    def _is_finger_up(self, lm: list, tip_idx: int, pip_idx: int, h: int) -> bool:
        """Return True when the finger tip is above its PIP joint (image coords)."""
        return lm[tip_idx].y * h < lm[pip_idx].y * h

    def _classify_gesture(self, lm: list, h: int) -> Gesture:
        """
        Rule-based gesture classifier (priority order):
          1. Eraser  — 4+ fingers up (open palm)
          2. Hover   — index + middle up, ring & pinky down
          3. Drawing — index up only
          4. None    — otherwise
        """
        index_up  = self._is_finger_up(lm, self._INDEX_TIP,  self._INDEX_PIP,  h)
        middle_up = self._is_finger_up(lm, self._MIDDLE_TIP, self._MIDDLE_PIP, h)
        ring_up   = self._is_finger_up(lm, self._RING_TIP,   self._RING_PIP,   h)
        pinky_up  = self._is_finger_up(lm, self._PINKY_TIP,  self._PINKY_PIP,  h)

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        if fingers_up >= 4:
            return Gesture.ERASER
        if index_up and middle_up and not ring_up and not pinky_up:
            return Gesture.HOVER
        if index_up and not middle_up:
            return Gesture.DRAWING
        return Gesture.NONE

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        lm: list,
        w: int,
        h: int,
    ) -> None:
        """Draw a minimal hand skeleton onto *frame* in-place."""
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]

        # Connections
        for a, b in self._CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 120), 1, cv2.LINE_AA)

        # Landmark dots
        for i, pt in enumerate(pts):
            # Fingertips slightly larger
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r, (0, 140, 80), 1, cv2.LINE_AA)
