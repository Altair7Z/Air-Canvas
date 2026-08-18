"""
canvas_engine.py
----------------
Manages the virtual drawing canvas as a transparent BGRA layer.
Handles stroke rendering, erasing, undo history, and compositing
the drawing layer onto the live webcam feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class BrushStyle:
    """Encapsulates the active brush configuration."""
    color: tuple[int, int, int] = (0, 0, 255)   # BGR
    size: int = 8
    eraser_size: int = 40


class CanvasEngine:
    """
    Maintains a transparent BGRA drawing canvas the same size as the
    webcam frame. Exposes methods to draw strokes, erase, and composite
    the drawing layer onto a BGR camera frame.

    Parameters
    ----------
    width : int
    height : int
    max_undo : int
        Maximum number of undo snapshots kept in memory.
    """

    def __init__(self, width: int, height: int, max_undo: int = 20) -> None:
        self._w = width
        self._h = height
        self._max_undo = max_undo

        # BGRA canvas — alpha=0 means transparent
        self._canvas: np.ndarray = np.zeros((height, width, 4), dtype=np.uint8)

        # Undo stack — stores compressed PNG snapshots for low memory use
        self._undo_stack: list[bytes] = []

        # Previous stroke point (for line continuity between frames)
        self._prev_point: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public API — Drawing
    # ------------------------------------------------------------------

    def begin_stroke(self, point: tuple[int, int]) -> None:
        """Call when a new drawing gesture starts (finger goes down)."""
        self._prev_point = point

    def end_stroke(self) -> None:
        """Call when the drawing gesture ends."""
        self._prev_point = None

    def draw_stroke(
        self,
        point: tuple[int, int],
        brush: BrushStyle,
    ) -> None:
        """
        Draw a continuous anti-aliased line from the previous position
        to *point*, then update the stored previous position.
        """
        if self._prev_point is None:
            self._prev_point = point
            return

        # Convert color to BGRA (full opacity)
        bgra = (*brush.color, 255)

        cv2.line(
            self._canvas,
            self._prev_point,
            point,
            bgra,
            brush.size,
            lineType=cv2.LINE_AA,
        )
        # Also draw a filled circle at the tip for smooth round caps
        cv2.circle(self._canvas, point, brush.size // 2, bgra, -1, cv2.LINE_AA)

        self._prev_point = point

    def erase_at(self, point: tuple[int, int], eraser_size: int) -> None:
        """Erase a circular region around *point* on the canvas."""
        x, y = point
        r = eraser_size // 2
        cv2.circle(self._canvas, (x, y), r, (0, 0, 0, 0), -1)
        self._prev_point = None

    def clear_canvas(self) -> None:
        """Push an undo snapshot then wipe the canvas to fully transparent."""
        self._push_undo()
        self._canvas[:] = 0

    def undo(self) -> None:
        """Restore the previous canvas state from the undo stack."""
        if self._undo_stack:
            data = self._undo_stack.pop()
            buf = np.frombuffer(data, dtype=np.uint8)
            self._canvas = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    # ------------------------------------------------------------------
    # Public API — Compositing
    # ------------------------------------------------------------------

    def composite(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Blend the drawing canvas (BGRA) on top of *bgr_frame* (BGR).

        Uses alpha compositing so transparent canvas pixels show the
        webcam feed underneath.

        Returns
        -------
        np.ndarray
            Composited BGR image ready to display.
        """
        # Extract alpha mask and normalise to [0, 1]
        alpha = self._canvas[:, :, 3:4].astype(np.float32) / 255.0
        canvas_bgr = self._canvas[:, :, :3].astype(np.float32)
        frame_f = bgr_frame.astype(np.float32)

        blended = frame_f * (1.0 - alpha) + canvas_bgr * alpha
        return blended.astype(np.uint8)

    # ------------------------------------------------------------------
    # Public API — Export
    # ------------------------------------------------------------------

    def get_canvas_bgra(self) -> np.ndarray:
        """Return the raw BGRA canvas (e.g. for saving)."""
        return self._canvas.copy()

    def get_drawing_on_white(self) -> np.ndarray:
        """Return the drawing composited onto a white background (BGR)."""
        white = np.full((self._h, self._w, 3), 255, dtype=np.uint8)
        return self.composite(white)

    @property
    def is_empty(self) -> bool:
        """True when the canvas has no visible pixels."""
        return not np.any(self._canvas[:, :, 3])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        """Compress the current canvas and push it onto the undo stack."""
        if len(self._undo_stack) >= self._max_undo:
            self._undo_stack.pop(0)  # Drop oldest
        success, buf = cv2.imencode(".png", self._canvas)
        if success:
            self._undo_stack.append(buf.tobytes())
