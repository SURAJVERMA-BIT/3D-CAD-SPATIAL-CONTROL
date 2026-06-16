# 3D CAD Spatial Control

Control 3D models in mid-air with your bare hands using just a webcam.
Powered by MediaPipe hand tracking and OpenCV.

The project ships three apps, from a simple mouse driver to a full holographic
manipulator.

## Apps

| Script | What it does |
|--------|--------------|
| [`src/holographic_control.py`](src/holographic_control.py) | **Holographic control.** Renders a 3D wireframe model over your camera feed and lets you rotate, scale and spin it directly with one hand. Self-contained — no other software needed. Can load your own `.obj` CAD models. |
| [`src/3d_model_controlV2.py`](src/3d_model_controlV2.py) | Drives the OS mouse from hand gestures (pinch-drag to grab, two hands to zoom) to control a model inside real CAD software. Uses the MediaPipe Tasks API (Python 3.13 compatible). |
| [`src/3D_model_control.py`](src/3D_model_control.py) | Original prototype (legacy MediaPipe `solutions` API + pyautogui). |

## Project layout

```
.
├── models/                     # MediaPipe model assets
│   └── hand_landmarker.task
├── src/                        # Application code
│   ├── holographic_control.py
│   ├── 3d_model_controlV2.py
│   └── 3D_model_control.py
├── tests/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.13 (the newer apps use the MediaPipe Tasks API)
- See [`requirements.txt`](requirements.txt)

## Installation

```bash
git clone https://github.com/SURAJVERMA-BIT/gesture-control.git
cd "3D cad SPATIAL CONTROL"
pip install -r requirements.txt
```

The `models/hand_landmarker.task` bundle is included in the repo, so the apps
run out of the box.

## Usage

### Holographic manipulator (recommended)

```bash
python src/holographic_control.py
```

One hand controls everything:

| Hand pose | Action |
|-----------|--------|
| Open hand, moving | Rotate the hologram (follows your palm; flick to spin with momentum) |
| Thumb + index only | Scale — spread to enlarge, pinch to shrink |
| Fist | Freeze / hold the orientation |
| No hand | Idle auto-spin |

Keys: `n` next model · `r` reset · `g` toggle hand overlay · `q` quit

Load your own CAD model (exported to `.obj`):

```bash
python src/holographic_control.py path/to/model.obj
```

### Mouse control (drive external CAD software)

```bash
python src/3d_model_controlV2.py
```

- Pinch (thumb + index) and move to grab/drag
- Two hands, move apart/together to zoom
- Press `q` in the window to quit
