/*
  Sign Interpreter (Demo)
  - Webcam + MediaPipe HandLandmarker via CDN
  - Counts extended fingers (0–5) with simple heuristics
  - Smoothing + optional text-to-speech
*/

import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const labelEl = document.getElementById("label");
const ttsToggle = document.getElementById("ttsToggle");

let handLandmarker = null;
let drawingUtils = null;
let videoStream = null;
let rafId = null;

class LabelSmoother {
  constructor(windowSize = 8) {
    this.windowSize = windowSize;
    this.buffer = [];
    this.current = null;
  }

  push(label) {
    this.buffer.push(label);
    if (this.buffer.length > this.windowSize) {
      this.buffer.shift();
    }
    return this.mode();
  }

  mode() {
    const counts = new Map();
    for (const entry of this.buffer) {
      if (entry === null || entry === undefined) continue;
      counts.set(entry, (counts.get(entry) || 0) + 1);
    }
    let best = null;
    let bestCount = 0;
    for (const [k, v] of counts.entries()) {
      if (v > bestCount) {
        best = k;
        bestCount = v;
      }
    }
    return best;
  }
}

const smoother = new LabelSmoother(10);
let lastSpoken = null;

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  try {
    statusEl.textContent = "Loading model...";
    await ensureModel();
    statusEl.textContent = "Requesting camera...";
    await startCamera();
    statusEl.textContent = "Detecting...";
    startLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message || err}`;
    startBtn.disabled = false;
  }
});

async function ensureModel() {
  if (handLandmarker) return;
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
  drawingUtils = new DrawingUtils(ctx);
}

async function startCamera() {
  if (videoStream) return;
  const constraints = {
    audio: false,
    video: { width: { ideal: 800 }, height: { ideal: 600 }, facingMode: "user" },
  };
  videoStream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = videoStream;
  await video.play();
  await new Promise((resolve) => {
    if (video.readyState >= 2) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  sizeCanvasToVideo();
}

function sizeCanvasToVideo() {
  const { videoWidth, videoHeight } = video;
  canvas.width = videoWidth;
  canvas.height = videoHeight;
}

function startLoop() {
  if (rafId) cancelAnimationFrame(rafId);
  const loop = () => {
    const nowMs = performance.now();
    const result = handLandmarker.detectForVideo(video, nowMs);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let label = null;
    if (result && result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      const handedness = result.handednesses?.[0]?.[0]?.categoryName || null;

      // Draw landmarks
      try {
        drawingUtils.drawLandmarks(landmarks, { color: "#22c55e", lineWidth: 2, radius: 2.2 });
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: "#38bdf8",
          lineWidth: 2,
        });
      } catch {}

      const count = countExtendedFingers(landmarks, handedness);
      label = Number.isFinite(count) ? `${count}` : null;
    }

    const smooth = smoother.push(label);
    if (smooth !== undefined && smooth !== null) {
      updateLabel(smooth);
    } else {
      updateLabel("None");
    }

    rafId = requestAnimationFrame(loop);
  };
  rafId = requestAnimationFrame(loop);
}

function updateLabel(value) {
  const prev = labelEl.dataset.value;
  if (`${value}` === prev) return;
  labelEl.dataset.value = `${value}`;
  labelEl.textContent = `${value}`;

  if (ttsToggle.checked) {
    speakValue(value);
  }
}

function speakValue(value) {
  const phrase = numberToWords(value);
  if (!phrase) return;
  if (lastSpoken === phrase) return;
  lastSpoken = phrase;
  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(phrase);
    utter.rate = 1.0;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
  } catch {}
}

function numberToWords(val) {
  const map = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
  };
  return map[String(val)] || null;
}

// Heuristic finger counter
function countExtendedFingers(landmarks, handedness) {
  if (!landmarks || landmarks.length < 21) return null;

  const idx = {
    WRIST: 0,
    THUMB_CMC: 1,
    THUMB_MCP: 2,
    THUMB_IP: 3,
    THUMB_TIP: 4,
    INDEX_MCP: 5,
    INDEX_PIP: 6,
    INDEX_DIP: 7,
    INDEX_TIP: 8,
    MIDDLE_MCP: 9,
    MIDDLE_PIP: 10,
    MIDDLE_DIP: 11,
    MIDDLE_TIP: 12,
    RING_MCP: 13,
    RING_PIP: 14,
    RING_DIP: 15,
    RING_TIP: 16,
    PINKY_MCP: 17,
    PINKY_PIP: 18,
    PINKY_DIP: 19,
    PINKY_TIP: 20,
  };

  const isFingerExtended = (tip, pip) => {
    // y is top=0 bottom=1 in normalized coords; tip above pip => extended
    const tipY = landmarks[tip].y;
    const pipY = landmarks[pip].y;
    return tipY + 0.02 < pipY; // small margin for noise
  };

  let count = 0;
  if (isFingerExtended(idx.INDEX_TIP, idx.INDEX_PIP)) count++;
  if (isFingerExtended(idx.MIDDLE_TIP, idx.MIDDLE_PIP)) count++;
  if (isFingerExtended(idx.RING_TIP, idx.RING_PIP)) count++;
  if (isFingerExtended(idx.PINKY_TIP, idx.PINKY_PIP)) count++;

  // Thumb: compare x position of tip vs IP depending on handedness
  const tipX = landmarks[idx.THUMB_TIP].x;
  const ipX = landmarks[idx.THUMB_IP].x;
  if (handedness === "Right") {
    if (tipX + 0.03 < ipX) count++; // thumb points left when extended
  } else if (handedness === "Left") {
    if (tipX - 0.03 > ipX) count++; // thumb points right when extended
  } else {
    // Fallback if handedness unknown: use absolute delta
    if (Math.abs(tipX - ipX) > 0.06) count++;
  }

  return count;
}

window.addEventListener("beforeunload", () => {
  if (rafId) cancelAnimationFrame(rafId);
  if (videoStream) {
    for (const track of videoStream.getTracks()) track.stop();
  }
  try { window.speechSynthesis.cancel(); } catch {}
});

// Handle resize: keep canvas in sync
window.addEventListener("resize", () => {
  if (!video || video.readyState < 2) return;
  sizeCanvasToVideo();
});
