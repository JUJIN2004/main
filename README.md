# Sign Interpreter (Demo)

A minimal client-side website that uses the webcam and MediaPipe Hands to detect a single hand and estimate how many fingers (0–5) are extended. Includes optional text-to-speech.

## Run locally

Serve the folder with any static server:

```bash
# Python
python3 -m http.server 5173

# Node
npx serve . -l 5173 --single
```

Open `http://localhost:5173` in your browser. If you open `index.html` from the filesystem, some browsers block ESM imports; prefer serving over HTTP.

## Features

- Realtime hand landmarks via MediaPipe Tasks (WebAssembly)
- Heuristic 0–5 finger counting with simple smoothing
- Optional text-to-speech output
- Fully client-side, no backend

## Notes and limitations

- This is NOT a full ASL interpreter; it only counts fingers.
- Results vary with lighting, background, camera angle, and hand orientation.
- Thumb detection can fail for unusual poses or rotations.

## Extend

Modify `app.js` to add gestures:

- Normalize landmarks (translate by wrist, scale by palm) and compare vectors via kNN.
- Add temporal logic for dynamic signs.

## Credits

- MediaPipe Tasks: Hand Landmarker
