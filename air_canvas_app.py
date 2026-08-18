"""
air_canvas_app.py
-----------------
Top-level application class that wires together HandTracker, CanvasEngine,
and UIOverlay into the main real-time loop.

Run directly:
    python air_canvas_app.py

Controls
--------
  Index finger only     → Draw
  Index + Middle finger → Select / hover (header palette interaction)
  Open palm (4+ up)     → Dynamic eraser
  S key                 → Save canvas
  U key / Ctrl+Z        → Undo last stroke
  Q / Esc               → Quit
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from canvas_engine import BrushStyle, CanvasEngine
from hand_tracker import Gesture, HandTracker
from ui_overlay import UIOverlay


class AirCanvasApp:
    """
    Main application class for Air Canvas.

    Responsible for:
    - Capturing and mirroring the webcam feed.
    - Orchestrating the hand tracker → gesture → canvas state machine.
    - Handling header palette interactions (colour, brush, clear, save, undo).
    - Maintaining a stable FPS measurement.
    - Exporting drawings as timestamped PNG files.

    Parameters
    ----------
    camera_index : int
        OpenCV camera device index (default 0).
    target_fps : int
        Desired frame rate; sets ``CAP_PROP_FPS`` if supported.
    save_dir : str or Path
        Directory where exported PNGs are written.
    """

    # Debounce interval (seconds) for palette button interactions
    _BUTTON_DEBOUNCE: float = 0.6

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 30,
        save_dir: str | Path = ".",
    ) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # ---- Camera setup -----------------------------------------------
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback without backend hint
            self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Ensure your webcam is connected and not in use."
            )

        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Read one frame to learn actual resolution
        ok, probe = self._cap.read()
        if not ok:
            raise RuntimeError("Camera opened but failed to grab the first frame.")
        self._h, self._w = probe.shape[:2]

        # ---- Sub-systems ------------------------------------------------
        self._tracker = HandTracker(ema_alpha=0.35)
        self._canvas = CanvasEngine(self._w, self._h)
        self._ui = UIOverlay(self._w, self._h)

        # ---- Brush state ------------------------------------------------
        default_color = UIOverlay.PALETTE_COLORS[0][1]  # Red
        self._brush = BrushStyle(color=default_color, size=10, eraser_size=50)

        # ---- Application state -----------------------------------------
        self._drawing: bool = False           # Whether currently in a draw stroke
        self._prev_gesture: Gesture = Gesture.NONE
        self._status_msg: str = ""
        self._status_expires: float = 0.0
        self._btn_last_hit: float = 0.0       # Debounce timestamp

        # FPS tracking
        self._fps: float = 0.0
        self._fps_alpha: float = 0.1           # EMA factor for FPS display
        self._last_ts: float = time.perf_counter()

        # Save flag (can be set by 'S' keypress as well as gesture)
        self._save_requested: bool = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main loop. Blocks until the user quits."""
        print(
            f"[AirCanvas] Running — resolution {self._w}×{self._h}\n"
            "  ✏  Index finger  → Draw\n"
            "  ☝  Index+Middle  → Select palette\n"
            "  ✋  Open palm     → Erase\n"
            "  S                → Save  |  U → Undo  |  Q/Esc → Quit\n"
        )

        cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Air Canvas", self._w, self._h)

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):   # Q or Esc
                    break
                if key == ord("s"):
                    self._save_requested = True
                if key == ord("u"):
                    self._canvas.undo()
                    self._set_status("Undo ✓", (180, 160, 80))

                ok, raw_frame = self._cap.read()
                if not ok:
                    print("[AirCanvas] Frame grab failed — skipping.")
                    continue

                # Mirror for natural interaction
                frame = cv2.flip(raw_frame, 1)

                # --- Hand tracking
                landmarks, annotated = self._tracker.process_frame(frame)

                # --- Update canvas state machine
                self._update_state(landmarks)

                # --- Composite: drawing layer on webcam feed
                display = self._canvas.composite(annotated)

                # --- Render UI on top
                self._render_ui(display, landmarks)

                # --- FPS
                now = time.perf_counter()
                dt = now - self._last_ts
                self._last_ts = now
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                self._fps = (
                    self._fps_alpha * inst_fps + (1 - self._fps_alpha) * self._fps
                )

                self._ui.draw_fps(display, self._fps)

                # --- Status message
                if time.time() < self._status_expires:
                    self._ui.draw_status_bar(display, self._status_msg)

                # --- Save if requested
                if self._save_requested:
                    self._save_canvas()
                    self._save_requested = False

                cv2.imshow("Air Canvas", display)

        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_state(self, landmarks) -> None:
        """
        Core gesture → canvas state machine.

        Called each frame after hand tracking.
        """
        if landmarks is None:
            # No hand — end any active stroke
            if self._drawing:
                self._canvas.end_stroke()
                self._drawing = False
            self._prev_gesture = Gesture.NONE
            return

        gesture = landmarks.gesture
        tip = landmarks.smooth_tip

        # ---- Eraser mode ------------------------------------------------
        if gesture == Gesture.ERASER:
            if self._drawing:
                self._canvas.end_stroke()
                self._drawing = False
            self._canvas.erase_at(tip, self._brush.eraser_size)

        # ---- Hover / selection mode ------------------------------------
        elif gesture == Gesture.HOVER:
            if self._drawing:
                self._canvas.end_stroke()
                self._drawing = False

            # Only process palette hits with debounce
            if self._ui.in_header(tip):
                now = time.time()
                if now - self._btn_last_hit > self._BUTTON_DEBOUNCE:
                    self._handle_palette_hit(tip, now)

        # ---- Drawing mode ----------------------------------------------
        elif gesture == Gesture.DRAWING:
            if self._ui.in_header(tip):
                # Finger inside header while drawing → pause stroke
                if self._drawing:
                    self._canvas.end_stroke()
                    self._drawing = False
            else:
                if not self._drawing:
                    self._canvas.begin_stroke(tip)
                    self._drawing = True
                self._canvas.draw_stroke(tip, self._brush)

        # ---- No recognised gesture ------------------------------------
        else:
            if self._drawing:
                self._canvas.end_stroke()
                self._drawing = False

        self._prev_gesture = gesture

    # ------------------------------------------------------------------
    # Palette interaction
    # ------------------------------------------------------------------

    def _handle_palette_hit(self, tip: tuple[int, int], now: float) -> None:
        """Process a finger hover over the header palette with debounce."""
        if (color := self._ui.get_color_hit(tip)) is not None:
            self._brush.color = color
            self._btn_last_hit = now
            return

        if (size := self._ui.get_brush_hit(tip)) is not None:
            self._brush.size = size
            self._btn_last_hit = now
            return

        if self._ui.is_clear_hit(tip):
            self._canvas.clear_canvas()
            self._set_status("Canvas cleared", (80, 80, 200))
            self._btn_last_hit = now
            return

        if self._ui.is_save_hit(tip):
            self._save_requested = True
            self._btn_last_hit = now
            return

        if self._ui.is_undo_hit(tip):
            self._canvas.undo()
            self._set_status("Undo ✓", (180, 160, 80))
            self._btn_last_hit = now
            return

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_ui(self, frame: np.ndarray, landmarks) -> None:
        """Draw the header palette and cursor indicator onto *frame*."""
        self._ui.draw_header(frame, self._brush.color, self._brush.size)

        if landmarks is not None:
            gesture_str = {
                Gesture.DRAWING: "DRAW",
                Gesture.HOVER:   "HOVER",
                Gesture.ERASER:  "ERASE",
                Gesture.NONE:    "NONE",
            }.get(landmarks.gesture, "NONE")

            self._ui.draw_cursor(
                frame,
                landmarks.smooth_tip,
                self._brush,
                gesture_str,
                dt=1.0 / max(self._fps, 1.0),
            )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_canvas(self) -> None:
        """Export the current drawing as a timestamped PNG."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._save_dir / f"air_canvas_{ts}.png"
        img = self._canvas.get_drawing_on_white()
        ok = cv2.imwrite(str(filename), img)
        if ok:
            msg = f"Saved → {filename.name}"
            color = (60, 200, 60)
            print(f"[AirCanvas] {msg}")
        else:
            msg = "Save failed!"
            color = (0, 0, 220)
        self._set_status(msg, color)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(
        self,
        msg: str,
        color: tuple[int, int, int] = (200, 200, 200),
        duration: float = 2.5,
    ) -> None:
        self._status_msg = msg
        self._status_expires = time.time() + duration

    def _cleanup(self) -> None:
        self._cap.release()
        self._tracker.release()
        cv2.destroyAllWindows()
        print("[AirCanvas] Session ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse minimal CLI args and start Air Canvas."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Air Canvas — draw in the air with your webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target frame rate.")
    parser.add_argument("--save-dir", type=str, default=".",
                        help="Directory for saved PNG exports.")
    args = parser.parse_args()

    try:
        app = AirCanvasApp(
            camera_index=args.camera,
            target_fps=args.fps,
            save_dir=args.save_dir,
        )
        app.run()
    except RuntimeError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[AirCanvas] Interrupted by user.")


if __name__ == "__main__":
    main()
