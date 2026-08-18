"""
ui_overlay.py
-------------
Renders the interactive top-palette header (color swatches, brush sizes,
clear/save buttons) directly onto a BGR frame using OpenCV drawing.

Also draws the animated cursor halo/ring around the active fingertip.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from canvas_engine import BrushStyle


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UIButton:
    """Represents a single clickable region in the header palette."""
    x: int
    y: int
    w: int
    h: int
    label: str
    color: Optional[tuple[int, int, int]] = None   # BGR fill, or None for text-only

    def contains(self, point: tuple[int, int]) -> bool:
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


# ---------------------------------------------------------------------------
# UIOverlay
# ---------------------------------------------------------------------------

class UIOverlay:
    """
    Draws the floating header palette and cursor indicator onto BGR frames.

    The palette is rendered on a semi-transparent dark strip at the top of
    the frame.  All interactive regions are exposed as ``UIButton`` objects
    so that ``AirCanvasApp`` can do hit-testing without coupling to the
    rendering internals.

    Parameters
    ----------
    frame_width : int
    frame_height : int
    header_height : int
        Pixel height of the top palette strip.
    """

    # Palette colour definitions (label → BGR)
    PALETTE_COLORS: list[tuple[str, tuple[int, int, int]]] = [
        ("Red",    (0,   0,   220)),
        ("Orange", (0,   140, 255)),
        ("Yellow", (0,   220, 220)),
        ("Green",  (0,   200, 0)),
        ("Cyan",   (200, 200, 0)),
        ("Blue",   (220, 80,  0)),
        ("Violet", (200, 0,   200)),
        ("White",  (255, 255, 255)),
    ]

    BRUSH_SIZES: list[tuple[str, int]] = [
        ("S", 4),
        ("M", 10),
        ("L", 20),
        ("XL", 32),
    ]

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        header_height: int = 90,
    ) -> None:
        self._fw = frame_width
        self._fh = frame_height
        self._hh = header_height

        # Built once; referenced for hit-testing
        self._color_buttons: list[UIButton] = []
        self._brush_buttons: list[UIButton] = []
        self._clear_btn: Optional[UIButton] = None
        self._save_btn: Optional[UIButton] = None
        self._undo_btn: Optional[UIButton] = None

        self._build_layout()

        # Animation state for cursor halo
        self._halo_phase: float = 0.0

    # ------------------------------------------------------------------
    # Layout builder
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Compute all button bounding boxes based on frame dimensions."""
        swatch_w = 52
        swatch_h = 42
        swatch_margin = 6
        swatch_top = 8

        # ---- Colour swatches (left side) --------------------------------
        x = 10
        for label, color in self.PALETTE_COLORS:
            btn = UIButton(
                x=x, y=swatch_top, w=swatch_w, h=swatch_h,
                label=label, color=color,
            )
            self._color_buttons.append(btn)
            x += swatch_w + swatch_margin

        # ---- Brush size toggles -----------------------------------------
        brush_w = 48
        brush_h = 42
        bx = x + 16
        for label, size in self.BRUSH_SIZES:
            btn = UIButton(
                x=bx, y=swatch_top, w=brush_w, h=brush_h,
                label=label, color=None,
            )
            self._brush_buttons.append(btn)
            bx += brush_w + swatch_margin

        # ---- Action buttons (right side) --------------------------------
        action_w = 70
        action_h = 42
        right_margin = 10
        ax = self._fw - right_margin - action_w

        self._save_btn = UIButton(
            x=ax, y=swatch_top, w=action_w, h=action_h,
            label="Save", color=(50, 180, 50),
        )
        ax -= action_w + swatch_margin

        self._clear_btn = UIButton(
            x=ax, y=swatch_top, w=action_w, h=action_h,
            label="Clear", color=(30, 30, 180),
        )
        ax -= action_w + swatch_margin

        self._undo_btn = UIButton(
            x=ax, y=swatch_top, w=action_w, h=action_h,
            label="Undo", color=(100, 100, 30),
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw_header(
        self,
        frame: np.ndarray,
        active_color: tuple[int, int, int],
        active_brush_size: int,
    ) -> None:
        """
        Render the floating palette onto *frame* (in-place).

        Parameters
        ----------
        frame : np.ndarray  BGR
        active_color : tuple[int, int, int]  currently selected colour (BGR)
        active_brush_size : int             currently selected brush size
        """
        # Semi-transparent dark background strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._fw, self._hh), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Thin separator line at the bottom of the header
        cv2.line(frame, (0, self._hh), (self._fw, self._hh), (60, 60, 60), 1)

        # ---- Draw colour swatches
        for btn in self._color_buttons:
            is_active = btn.color == active_color
            self._draw_swatch(frame, btn, is_active)

        # ---- Brush size buttons
        for btn, (_, size) in zip(self._brush_buttons, self.BRUSH_SIZES):
            is_active = size == active_brush_size
            self._draw_brush_btn(frame, btn, size, is_active, active_color)

        # ---- Action buttons
        for action_btn in [self._clear_btn, self._save_btn, self._undo_btn]:
            if action_btn:
                self._draw_action_btn(frame, action_btn)

        # ---- Section labels (tiny)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "COLOR", (10, self._hh - 6), font, 0.32,
                    (120, 120, 120), 1, cv2.LINE_AA)
        bx0 = self._brush_buttons[0].x
        cv2.putText(frame, "BRUSH", (bx0, self._hh - 6), font, 0.32,
                    (120, 120, 120), 1, cv2.LINE_AA)

    def draw_cursor(
        self,
        frame: np.ndarray,
        tip: tuple[int, int],
        brush: BrushStyle,
        gesture_label: str,
        dt: float = 0.033,
    ) -> None:
        """
        Draw the animated halo/ring cursor around the active fingertip.

        Parameters
        ----------
        frame : np.ndarray
        tip : tuple[int, int]           smoothed fingertip position
        brush : BrushStyle
        gesture_label : str             e.g. 'DRAW', 'HOVER', 'ERASE'
        dt : float                      seconds since last frame (for animation)
        """
        self._halo_phase = (self._halo_phase + dt * 3.0) % (2 * math.pi)
        pulse = 1.0 + 0.15 * math.sin(self._halo_phase)

        color = brush.color

        if gesture_label == "ERASE":
            # White dashed circle for eraser
            r = int(brush.eraser_size * pulse * 0.5)
            self._draw_dashed_circle(frame, tip, r, (200, 200, 200), 2)
            # Inner X cross
            d = r // 2
            x, y = tip
            cv2.line(frame, (x - d, y - d), (x + d, y + d), (150, 150, 150), 1)
            cv2.line(frame, (x + d, y - d), (x - d, y + d), (150, 150, 150), 1)
        elif gesture_label == "HOVER":
            # Hollow double ring in muted grey
            r = int(14 * pulse)
            cv2.circle(frame, tip, r, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.circle(frame, tip, r + 5, (100, 100, 100), 1, cv2.LINE_AA)
        else:
            # Drawing mode — solid inner dot + outer halo matching brush colour
            r_inner = max(2, brush.size // 2)
            r_outer = int((brush.size + 10) * pulse * 0.5)
            cv2.circle(frame, tip, r_inner, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tip, r_outer, color, 1, cv2.LINE_AA)
            # Soft glow (low-alpha outer ring)
            glow = frame.copy()
            cv2.circle(glow, tip, r_outer + 4, color, 2, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

        # Gesture label badge
        label_map = {"DRAW": "✏ DRAW", "HOVER": "☝ SELECT", "ERASE": "⬜ ERASE", "NONE": ""}
        label_text = label_map.get(gesture_label, gesture_label)
        if label_text:
            x, y = tip
            tx = max(4, x - 30)
            ty = max(self._hh + 14, y - 18)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_text, font, 0.38, 1)
            bg_pt1 = (tx - 3, ty - th - 3)
            bg_pt2 = (tx + tw + 3, ty + 3)
            cv2.rectangle(frame, bg_pt1, bg_pt2, (20, 20, 20), -1)
            cv2.putText(frame, label_text, (tx, ty), font, 0.38,
                        (230, 230, 230), 1, cv2.LINE_AA)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Render the FPS counter in the bottom-left corner."""
        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (10, self._fh - 12), font, 0.55,
                    (0, 200, 100), 1, cv2.LINE_AA)

    def draw_status_bar(
        self,
        frame: np.ndarray,
        message: str,
        color: tuple[int, int, int] = (200, 200, 200),
    ) -> None:
        """Show a temporary status message at the bottom-right."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(message, font, 0.45, 1)
        x = self._fw - tw - 12
        y = self._fh - 12
        cv2.putText(frame, message, (x, y), font, 0.45, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Hit-testing helpers (used by AirCanvasApp)
    # ------------------------------------------------------------------

    def get_color_hit(
        self, point: tuple[int, int]
    ) -> Optional[tuple[int, int, int]]:
        """Return the BGR colour of whichever swatch the point falls on."""
        for btn in self._color_buttons:
            if btn.contains(point):
                return btn.color
        return None

    def get_brush_hit(self, point: tuple[int, int]) -> Optional[int]:
        """Return the brush pixel size if a brush button was hit."""
        for btn, (_, size) in zip(self._brush_buttons, self.BRUSH_SIZES):
            if btn.contains(point):
                return size
        return None

    def is_clear_hit(self, point: tuple[int, int]) -> bool:
        return bool(self._clear_btn and self._clear_btn.contains(point))

    def is_save_hit(self, point: tuple[int, int]) -> bool:
        return bool(self._save_btn and self._save_btn.contains(point))

    def is_undo_hit(self, point: tuple[int, int]) -> bool:
        return bool(self._undo_btn and self._undo_btn.contains(point))

    def in_header(self, point: tuple[int, int]) -> bool:
        """True if *point* is inside the header palette zone."""
        return point[1] <= self._hh

    @property
    def header_height(self) -> int:
        return self._hh

    # ------------------------------------------------------------------
    # Private rendering primitives
    # ------------------------------------------------------------------

    def _draw_swatch(
        self,
        frame: np.ndarray,
        btn: UIButton,
        is_active: bool,
    ) -> None:
        r = 6  # corner radius
        color = btn.color  # type: ignore[assignment]
        x, y, w, h = btn.x, btn.y, btn.w, btn.h

        # Filled rounded rect (approximated with rectangle + circles)
        cv2.rectangle(frame, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(frame, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r),
                       (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(frame, (cx, cy), r, color, -1)

        if is_active:
            # Bright white border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # Tiny check mark
            cx, cy = btn.center
            cv2.putText(frame, "✓", (cx - 7, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_brush_btn(
        self,
        frame: np.ndarray,
        btn: UIButton,
        size: int,
        is_active: bool,
        active_color: tuple[int, int, int],
    ) -> None:
        x, y, w, h = btn.x, btn.y, btn.w, btn.h
        bg = (50, 50, 50) if not is_active else (70, 70, 70)
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
        if is_active:
            cv2.rectangle(frame, (x, y), (x + w, y + h), active_color, 2)

        # Preview dot scaled to brush size
        cx, cy = btn.center
        preview_r = max(2, min(size // 2, h // 3))
        dot_color = active_color if is_active else (160, 160, 160)
        cv2.circle(frame, (cx, cy - 4), preview_r, dot_color, -1, cv2.LINE_AA)

        # Label
        cv2.putText(frame, btn.label, (cx - 6, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220, 220, 220) if is_active else (120, 120, 120),
                    1, cv2.LINE_AA)

    def _draw_action_btn(self, frame: np.ndarray, btn: UIButton) -> None:
        x, y, w, h = btn.x, btn.y, btn.w, btn.h
        bg = btn.color or (60, 60, 60)
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)
        cx, cy = btn.center
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(btn.label, font, 0.42, 1)
        cv2.putText(frame, btn.label, (cx - tw // 2, cy + th // 2),
                    font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_circle(
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        thickness: int,
        num_dashes: int = 16,
    ) -> None:
        cx, cy = center
        step = 2 * math.pi / (num_dashes * 2)
        for i in range(0, num_dashes * 2, 2):
            a1 = i * step
            a2 = a1 + step
            pt1 = (int(cx + radius * math.cos(a1)), int(cy + radius * math.sin(a1)))
            pt2 = (int(cx + radius * math.cos(a2)), int(cy + radius * math.sin(a2)))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
