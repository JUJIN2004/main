/*
  Sign Language Interpreter (ASL)
  - Webcam + MediaPipe HandLandmarker via CDN
  - Recognizes ASL alphabet letters (A–Z) from hand landmarks
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
const subtitleEl = document.getElementById("subtitleText");
const clearSubsBtn = document.getElementById("clearSubsBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const switchCameraBtn = document.getElementById("switchCameraBtn");
const videoCanvasWrap = document.getElementById("videoCanvasWrap");

/** "user" = front, "environment" = back. Front uses mirrored video+skeleton; subtitle never mirrored. */
let currentFacingMode = "user";
let handLandmarker = null;
let subtitleTranscript = "";
let lastAppendedLetter = null;
let drawingUtils = null;
let videoStream = null;
let rafId = null;
let framesSinceLetter = 0;
const PAUSE_FRAMES_FOR_WORD = 90; // ~1.5 s at 60fps → treat as end of word

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

const smoother = new LabelSmoother(12);
let lastSpoken = null;

clearSubsBtn.addEventListener("click", () => {
  subtitleTranscript = "";
  lastAppendedLetter = null;
  if (subtitleEl) subtitleEl.textContent = "";
});

function deleteLastLetter() {
  if (subtitleTranscript.length === 0) return;
  subtitleTranscript = subtitleTranscript.slice(0, -1);
  lastAppendedLetter = null;
  if (subtitleEl) subtitleEl.textContent = subtitleTranscript;
}

backspaceBtn.addEventListener("click", deleteLastLetter);

document.addEventListener("keydown", (e) => {
  if (e.key === "Backspace" && !e.ctrlKey && !e.metaKey && !e.altKey) {
    const target = e.target;
    const isInput = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
    if (!isInput) {
      e.preventDefault();
      deleteLastLetter();
    }
  }
});

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

function stopCamera() {
  if (videoStream) {
    for (const track of videoStream.getTracks()) track.stop();
    videoStream = null;
  }
  video.srcObject = null;
}

function updateMirrorClass() {
  if (videoCanvasWrap) {
    if (currentFacingMode === "user") {
      videoCanvasWrap.classList.add("mirror");
    } else {
      videoCanvasWrap.classList.remove("mirror");
    }
  }
}

async function startCamera(facingMode = "user") {
  if (videoStream) {
    const currentTrack = videoStream.getVideoTracks()[0];
    const currentFacing = currentTrack?.getSettings?.()?.facingMode;
    if (currentFacing === facingMode) return;
    stopCamera();
  }
  const constraints = {
    audio: false,
    video: { width: { ideal: 800 }, height: { ideal: 600 }, facingMode },
  };
  videoStream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = videoStream;
  await video.play();
  await new Promise((resolve) => {
    if (video.readyState >= 2) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  currentFacingMode = facingMode;
  updateMirrorClass();
  sizeCanvasToVideo();
  if (switchCameraBtn) switchCameraBtn.disabled = false;
}

switchCameraBtn.addEventListener("click", async () => {
  if (!videoStream || !handLandmarker) return;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  stopCamera();
  const nextMode = currentFacingMode === "user" ? "environment" : "user";
  try {
    statusEl.textContent = "Switching camera...";
    await startCamera(nextMode);
    statusEl.textContent = "Detecting...";
    startLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message || err}`;
  }
});

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
      const rawLandmarks = result.landmarks[0];
      let handedness = result.handednesses?.[0]?.[0]?.categoryName || null;

      // Draw with raw landmarks so skeleton aligns with video (video is mirrored via CSS when front)
      try {
        drawingUtils.drawLandmarks(rawLandmarks, { color: "#22c55e", lineWidth: 2, radius: 2.2 });
        drawingUtils.drawConnectors(rawLandmarks, HandLandmarker.HAND_CONNECTIONS, {
          color: "#38bdf8",
          lineWidth: 2,
        });
      } catch {}

      // For front camera the display is mirrored; flip for recognition so letters match what user sees. Subtitle stays normal.
      let landmarksForRecognition = rawLandmarks;
      let handednessForRecognition = handedness;
      if (currentFacingMode === "user") {
        landmarksForRecognition = rawLandmarks.map((p) => ({ ...p, x: 1 - p.x }));
        handednessForRecognition = handedness === "Left" ? "Right" : handedness === "Right" ? "Left" : handedness;
      }
      label = recognizeASLLetter(landmarksForRecognition, handednessForRecognition);
    }

    const smooth = smoother.push(label);
    if (smooth !== undefined && smooth !== null && smooth !== "") {
      const isLetter = /^[A-Za-z]$/.test(String(smooth));
      if (isLetter) {
        framesSinceLetter = 0;
        updateLabel(smooth);
      } else {
        updateLabel("—");
        framesSinceLetter++;
      }
    } else {
      updateLabel("—");
      framesSinceLetter++;
    }

    // After pause with no hand, treat current segment as a word: speak it, then add space
    if (framesSinceLetter === PAUSE_FRAMES_FOR_WORD && subtitleTranscript.length > 0) {
      const trimmed = subtitleTranscript.trim();
      const lastSpace = trimmed.lastIndexOf(" ");
      const word = lastSpace < 0 ? trimmed : trimmed.slice(lastSpace + 1);
      if (word.length > 0) {
        speakWord(word);
        subtitleTranscript = trimmed + " ";
        if (subtitleEl) subtitleEl.textContent = subtitleTranscript;
      }
      framesSinceLetter = PAUSE_FRAMES_FOR_WORD + 1; // avoid flushing again until next word
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

  // Append new letter to subtitle when it changes (letters only, never numbers)
  const isLetter = value && /^[A-Za-z]$/.test(String(value));
  if (isLetter && value !== lastAppendedLetter) {
    lastAppendedLetter = value;
    subtitleTranscript += value.toUpperCase();
    if (subtitleEl) subtitleEl.textContent = subtitleTranscript;
  }
  if (!isLetter) lastAppendedLetter = null;

  // Don't speak letters one-by-one; words are spoken when you pause (see pause handler)
}

function speakWord(word) {
  if (!word || !ttsToggle.checked) return;
  const w = String(word).toLowerCase().trim();
  if (!w.length) return;
  if (lastSpoken === w) return;
  lastSpoken = w;
  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(w);
    utter.rate = 0.9;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
  } catch {}
}

// MediaPipe hand landmark indices
const IDX = {
  WRIST: 0, THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = (a.z || 0) - (b.z || 0);
  return Math.hypot(dx, dy, dz);
}

function isFingerExtended(landmarks, tipIdx, pipIdx) {
  const tipY = landmarks[tipIdx].y, pipY = landmarks[pipIdx].y;
  return tipY + 0.03 < pipY;
}

function isFingerCurled(landmarks, tipIdx, pipIdx) {
  const tipY = landmarks[tipIdx].y, pipY = landmarks[pipIdx].y;
  return tipY > pipY + 0.02;
}

function thumbOut(landmarks, handedness) {
  const tip = landmarks[IDX.THUMB_TIP], ip = landmarks[IDX.THUMB_IP];
  const dx = tip.x - ip.x;
  if (handedness === "Right") return dx < -0.04;
  if (handedness === "Left") return dx > 0.04;
  return Math.abs(dx) > 0.08;
}

function thumbTouchingIndex(landmarks) {
  return dist(landmarks[IDX.THUMB_TIP], landmarks[IDX.INDEX_TIP]) < 0.10;
}

function thumbTouchingMiddle(landmarks) {
  return dist(landmarks[IDX.THUMB_TIP], landmarks[IDX.MIDDLE_TIP]) < 0.1;
}

function thumbNearIndexPalm(landmarks) {
  const d = dist(landmarks[IDX.THUMB_TIP], landmarks[IDX.INDEX_MCP]);
  return d < 0.12;
}

function indexBent(landmarks) {
  const tip = landmarks[IDX.INDEX_TIP], pip = landmarks[IDX.INDEX_PIP];
  return tip.y > pip.y - 0.02;
}

/** Recognizes ASL letter from hand landmarks (single hand, palm roughly toward camera). */
function recognizeASLLetter(landmarks, handedness) {
  if (!landmarks || landmarks.length < 21) return null;

  const iExt = isFingerExtended(landmarks, IDX.INDEX_TIP, IDX.INDEX_PIP);
  const mExt = isFingerExtended(landmarks, IDX.MIDDLE_TIP, IDX.MIDDLE_PIP);
  const rExt = isFingerExtended(landmarks, IDX.RING_TIP, IDX.RING_PIP);
  const pExt = isFingerExtended(landmarks, IDX.PINKY_TIP, IDX.PINKY_PIP);
  const thumbOut_ = thumbOut(landmarks, handedness);
  const thumbTouchIdx = thumbTouchingIndex(landmarks);
  const thumbTouchMid = thumbTouchingMiddle(landmarks);
  const thumbNearIdx = thumbNearIndexPalm(landmarks);
  const idxBent = indexBent(landmarks);
  const extendedCount = [iExt, mExt, rExt, pExt].filter(Boolean).length;

  // O: thumb and index form circle (check before A — both have fingers closed, O has thumb touching index)
  if (!iExt && !mExt && !rExt && !pExt && thumbTouchIdx) return "O";

  // A: fist, thumb to side (all fingers closed, thumb not touching index)
  if (!iExt && !mExt && !rExt && !pExt && thumbOut_) return "A";

  // B: all four fingers up, thumb in
  if (iExt && mExt && rExt && pExt && !thumbOut_) return "B";

  // C: curved C – thumb and fingers curved, not fully closed
  const thumbCurved = !thumbOut_ && dist(landmarks[IDX.THUMB_TIP], landmarks[IDX.INDEX_TIP]) < 0.2;
  if (extendedCount <= 1 && thumbCurved && !iExt && !pExt) return "C";

  // D: index up, thumb touches middle, others closed
  if (iExt && !mExt && !rExt && !pExt && thumbTouchMid) return "D";

  // E: fingers curved over thumb (thumb tucked)
  if (!iExt && !mExt && !rExt && !pExt && !thumbOut_ && thumbNearIdx) return "E";

  // F: OK sign – thumb+index circle, middle/ring/pinky up
  if (thumbTouchIdx && mExt && rExt && pExt) return "F";

  // G: index pointing, thumb in (gun shape)
  if (iExt && !mExt && !rExt && !pExt && !thumbTouchMid && !thumbOut_) return "G";

  // I: pinky up, others closed
  if (!iExt && !mExt && !rExt && pExt && !thumbOut_) return "I";

  // K: thumb between index and middle, V shape
  if (iExt && mExt && !rExt && !pExt && thumbOut_) return "K";

  // L: index and thumb L
  if (iExt && !mExt && !rExt && !pExt && thumbOut_) return "L";

  // M: thumb under index/middle/ring (three fingers down)
  if (!iExt && !mExt && !rExt && pExt && !thumbOut_) return "M";

  // N: thumb under index and middle
  if (!iExt && !mExt && rExt && pExt && !thumbOut_) return "N";

  // R: index and middle crossed (fingers close together)
  if (iExt && mExt && !rExt && !pExt) {
    const midX = landmarks[IDX.MIDDLE_TIP].x, idxX = landmarks[IDX.INDEX_TIP].x;
    if (Math.abs(midX - idxX) < 0.05) return "R";
  }

  // S: fist (thumb in front of fingers)
  if (!iExt && !mExt && !rExt && !pExt && !thumbOut_ && !thumbTouchIdx) return "S";

  // T: thumb between index and middle, fingers closed
  if (!iExt && !mExt && !rExt && !pExt && thumbOut_ && thumbTouchMid) return "T";

  // U / H: index and middle up together. V: index and middle spread
  if (iExt && mExt && !rExt && !pExt && !thumbOut_) {
    const spread = Math.abs(landmarks[IDX.INDEX_TIP].x - landmarks[IDX.MIDDLE_TIP].x);
    if (spread > 0.05) return "V";
    return "U";
  }

  // W: index, middle, ring up (spread)
  if (iExt && mExt && rExt && !pExt && !thumbOut_) return "W";

  // X: index bent at PIP
  if (idxBent && !mExt && !rExt && !pExt && !thumbOut_) return "X";

  // Y: thumb and pinky out
  if (!iExt && !mExt && !rExt && pExt && thumbOut_) return "Y";

  return null;
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
