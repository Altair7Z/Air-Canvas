# ✋ Air Canvas — Draw in the Air with Your Webcam

Real-time virtual drawing application using **MediaPipe Hands** + **OpenCV**.
Point your index finger at the webcam and paint in mid-air — no mouse, no touch screen.

---

## ✨ Features

| Feature | Detail |
|---|---|
| ✏ **Draw** | Index finger up → smooth EMA-buffered strokes |
| ☝ **Select** | Index + Middle finger → hover over palette without drawing |
| ✋ **Erase** | 4–5 fingers up (open palm) → dynamic circle eraser |
| 🎨 **8 Colors** | Red · Orange · Yellow · Green · Cyan · Blue · Violet · White |
| 🖌 **4 Brush Sizes** | S (4px) · M (10px) · L (20px) · XL (32px) |
| ↩ **Undo** | Gesture-select Undo button or press `U` (up to 20 levels) |
| 💾 **Save** | Exports `air_canvas_YYYYMMDD_HHMMSS.png` on `S` or gesture |
| 📊 **FPS Counter** | Live EMA-smoothed FPS shown bottom-left |

---

## 🗂 Project Structure

```
air_canvas/
├── air_canvas_app.py   # AirCanvasApp — main loop & state machine
├── hand_tracker.py     # HandTracker  — MediaPipe wraper, EMA, gesture classifier
├── canvas_engine.py    # CanvasEngine — BGRA drawing canvas, undo, compositing
├── ui_overlay.py       # UIOverlay    — header palette, cursor halo, FPS
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run

```powershell
python air_canvas_app.py
```

#### Optional flags

```
--camera  INT   Camera device index (default 0)
--fps     INT   Target frame rate   (default 30)
--save-dir PATH Directory for PNG exports (default: current dir)
```

---

## ⌨ Keyboard Shortcuts

| Key | Action |
|---|---|
| `S` | Save canvas as PNG |
| `U` | Undo last stroke |
| `Q` or `Esc` | Quit |

---

## 🤚 Gesture Guide

```
Drawing  ─── Index finger up, all others down
              ↑ Traces smooth lines on the canvas

Hover    ─── Index + Middle fingers up
              ↑ Move over palette to change colour / brush / action

Erase    ─── Open palm (4-5 fingers up)
              ↑ Circular eraser follows fingertip
```

---

## 🏗 Architecture

```
AirCanvasApp
│
├─ HandTracker          (hand_tracker.py)
│   ├─ MediaPipe Hands  → 21 landmarks
│   ├─ EMA smoother     → jitter-free coordinates
│   └─ GestureClassifier → DRAWING / HOVER / ERASER / NONE
│
├─ CanvasEngine         (canvas_engine.py)
│   ├─ BGRA canvas      → alpha-composited over webcam feed
│   ├─ Stroke renderer  → anti-aliased lines + round caps
│   ├─ Eraser           → circular alpha-clearing
│   └─ Undo stack       → PNG-compressed snapshots
│
└─ UIOverlay            (ui_overlay.py)
    ├─ Header palette   → colour swatches, brush toggles, action buttons
    ├─ Cursor halo      → animated pulsing ring matching active tool
    └─ FPS / status     → real-time diagnostics
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `mediapipe ≥ 0.10` | Hand landmark detection (21 pts @ 30+ FPS) |
| `opencv-python ≥ 4.9` | Frame capture, rendering, PNG export |
| `numpy ≥ 1.24` | Array manipulation & canvas blending |

---

## 🔧 Troubleshooting

**"Cannot open camera"** — Try `--camera 1` or `--camera 2` if you have multiple cameras.

**Low FPS** — Reduce resolution in `air_canvas_app.py` (`CAP_PROP_FRAME_WIDTH/HEIGHT`), or lower `detection_confidence` in `HandTracker`.

**Jittery strokes** — Lower `ema_alpha` in `HandTracker` (e.g. `0.2`) for smoother but slightly laggier tracking.
