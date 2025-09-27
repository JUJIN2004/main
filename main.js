// Minimal client for webcam + overlay + MediaPipe HandLandmarker detection + heuristics

/** @type {HTMLVideoElement} */
const video = document.getElementById('video');
/** @type {HTMLCanvasElement} */
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const fpsEl = document.getElementById('fps');

const currentSignEl = document.getElementById('currentSign');
const sentenceEl = document.getElementById('sentence');
const btnAppend = document.getElementById('btnAppend');
const btnSpace = document.getElementById('btnSpace');
const btnBackspace = document.getElementById('btnBackspace');
const btnClear = document.getElementById('btnClear');
const btnCopy = document.getElementById('btnCopy');

let stream = null;
let running = false;
let lastFrameTime = performance.now();
let frames = 0;
let fps = 0;

// MediaPipe HandLandmarker
let handLandmarker = null;
let mpReady = false;
let lastVideoTimestamp = -1;
let lastDetections = [];

async function initHandLandmarker() {
  try {
    const { FilesetResolver, HandLandmarker } = window.MPTasks || {};
    if (!FilesetResolver || !HandLandmarker) return;
    const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm');
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
      },
      numHands: 2,
      runningMode: 'VIDEO'
    });
    mpReady = true;
  } catch (e) {
    console.error('Failed to init MediaPipe HandLandmarker', e);
  }
}

initHandLandmarker();

function updateFps() {
  const now = performance.now();
  frames += 1;
  if (now - lastFrameTime >= 1000) {
    fps = frames;
    frames = 0;
    lastFrameTime = now;
    fpsEl.textContent = String(fps);
  }
}

function resizeCanvasToVideo() {
  const { videoWidth, videoHeight } = video;
  if (videoWidth && videoHeight) {
    overlay.width = videoWidth;
    overlay.height = videoHeight;
  }
}

async function startCamera() {
  if (running) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
    resizeCanvasToVideo();
    running = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    enableSentenceControls(true);
    requestAnimationFrame(loop);
  } catch (err) {
    console.error('Failed to start camera', err);
    alert('Failed to access camera. Please allow camera permissions.');
  }
}

function stopCamera() {
  running = false;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  btnStart.disabled = false;
  btnStop.disabled = true;
}

function enableSentenceControls(enable) {
  btnAppend.disabled = !enable;
  btnSpace.disabled = !enable;
  btnBackspace.disabled = !enable;
  btnClear.disabled = !enable;
  btnCopy.disabled = !enable;
}

// Placeholder detection result
let currentSign = '—';

function drawOverlay() {
  if (!overlay.width || !overlay.height) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!lastDetections || lastDetections.length === 0) return;
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(106,162,255,0.9)';
  ctx.fillStyle = 'rgba(155,219,255,0.9)';

  const EDGES = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
  ];

  for (const lms of lastDetections) {
    // lines
    ctx.beginPath();
    for (const [a, b] of EDGES) {
      const pa = lms[a];
      const pb = lms[b];
      if (!pa || !pb) continue;
      const x1 = pa.x * overlay.width;
      const y1 = pa.y * overlay.height;
      const x2 = pb.x * overlay.width;
      const y2 = pb.y * overlay.height;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    // tips
    for (const idx of [4,8,12,16,20]) {
      const p = lms[idx];
      if (!p) continue;
      const x = p.x * overlay.width;
      const y = p.y * overlay.height;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function loop() {
  if (!running) return;
  updateFps();
  detectHands();
  drawOverlay();
  currentSignEl.textContent = currentSign;
  requestAnimationFrame(loop);
}

btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);

// Sentence assembly
btnAppend.addEventListener('click', () => {
  if (currentSign && currentSign !== '—') {
    sentenceEl.textContent += currentSign;
  }
});
btnSpace.addEventListener('click', () => {
  sentenceEl.textContent += ' ';
});
btnBackspace.addEventListener('click', () => {
  const text = sentenceEl.textContent || '';
  sentenceEl.textContent = text.slice(0, -1);
});
btnClear.addEventListener('click', () => {
  sentenceEl.textContent = '';
});
btnCopy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(sentenceEl.textContent || '');
  } catch (e) {
    console.warn('Clipboard failed', e);
  }
});

// Resize overlay when the video metadata loads
video.addEventListener('loadedmetadata', resizeCanvasToVideo);

// ---------------------------------
// Hand detection + heuristics
// ---------------------------------

function detectHands() {
  if (!mpReady || !handLandmarker) return;
  const ts = performance.now();
  if (video.paused || video.ended) return;
  // Avoid duplicate processing if timestamp unchanged
  if (ts === lastVideoTimestamp) return;
  lastVideoTimestamp = ts;
  try {
    const result = handLandmarker.detectForVideo(video, ts);
    const landmarksList = result?.landmarks || [];
    lastDetections = landmarksList;
    updateClassification(landmarksList, result?.handednesses || []);
  } catch (e) {
    // Ignore transient errors during startup
  }
}

function vector(p1, p2) {
  return { x: p2.x - p1.x, y: p2.y - p1.y };
}

function dot(v1, v2) { return v1.x * v2.x + v1.y * v2.y; }
function mag(v) { return Math.hypot(v.x, v.y); }
function angleDeg(a, b, c) {
  // angle at b between ba and bc
  const v1 = vector(b, a);
  const v2 = vector(b, c);
  const cos = dot(v1, v2) / (mag(v1) * mag(v2) + 1e-6);
  return Math.acos(Math.max(-1, Math.min(1, cos))) * (180 / Math.PI);
}

function isFingerExtended(lms, mcp, pip, dip, tip) {
  const ang = angleDeg(lms[mcp], lms[pip], lms[dip]);
  const ang2 = angleDeg(lms[pip], lms[dip], lms[tip]);
  // Extended if both joint angles are fairly straight
  return ang > 155 && ang2 > 155;
}

function isThumbExtended(lms) {
  // Use mcp=2, ip=3, tip=4, with cmc=1
  const ang = angleDeg(lms[1], lms[2], lms[4]);
  return ang > 150;
}

function classify(landmarks) {
  if (!landmarks || landmarks.length < 21) return '—';
  const lms = landmarks;
  const indexUp = isFingerExtended(lms, 5, 6, 7, 8);
  const middleUp = isFingerExtended(lms, 9, 10, 11, 12);
  const ringUp = isFingerExtended(lms, 13, 14, 15, 16);
  const pinkyUp = isFingerExtended(lms, 17, 18, 19, 20);
  const thumbUp = isThumbExtended(lms);

  const numExtended = [indexUp, middleUp, ringUp, pinkyUp].filter(Boolean).length + (thumbUp ? 1 : 0);

  // Thumbs up: thumb extended, others mostly folded
  if (thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) return 'GOOD';

  // Index only
  if (indexUp && !middleUp && !ringUp && !pinkyUp) return 'NO';

  // Fist: none extended
  if (!indexUp && !middleUp && !ringUp && !pinkyUp && !thumbUp) return 'YES';

  // Open palm: most or all extended
  if (numExtended >= 4) return 'HELLO';

  return '—';
}

const recent = [];
const RECENT_MAX = 12;
function updateClassification(landmarksList, handednesses) {
  let label = '—';
  if (landmarksList && landmarksList.length > 0) {
    // Use first hand for simplicity
    label = classify(landmarksList[0]);
  }
  recent.push(label);
  if (recent.length > RECENT_MAX) recent.shift();

  // Compute mode excluding '—'
  const counts = new Map();
  for (const l of recent) {
    if (l === '—') continue;
    counts.set(l, (counts.get(l) || 0) + 1);
  }
  let best = '—';
  let bestCount = 0;
  counts.forEach((c, k) => { if (c > bestCount) { best = k; bestCount = c; } });
  // Require at least 4 frames agreement
  currentSign = bestCount >= 4 ? best : '—';
}

