# Sign Language Interpreter (Browser Demo)

A minimal in-browser demo that uses your webcam and MediaPipe HandLandmarker to detect hand landmarks, applies simple heuristics to classify a tiny set of signs, and lets you build a sentence from recognized signs. All processing happens locally in your browser.

## Supported Demo Signs
- HELLO: open palm facing camera (most fingers extended)
- YES: closed fist (no fingers extended)
- NO: index finger up (only index extended)
- GOOD: thumbs up (thumb extended, other fingers folded)

Heuristics are simplistic and for demonstration only. For robust recognition, train a model on labeled sign sequences.

## Run Locally
Camera access requires a secure origin. Localhost is allowed by browsers.

1. Start a static server in this folder:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory /workspace
```

2. Open `http://127.0.0.1:8000` and allow camera permissions.

3. Click "Start Camera" and try the demo signs.

## Tech
- MediaPipe Tasks Vision `HandLandmarker` via ESM CDN
- Plain HTML/CSS/JS, no build step

## Notes
- Lighting and background significantly affect detection quality.
- The heuristics may misclassify partial poses or occlusions.
- Extend this by collecting data and training a temporal classifier (e.g., TF.js or on-device TFLite) to recognize a broader ASL subset and multi-hand, multi-frame gestures.
