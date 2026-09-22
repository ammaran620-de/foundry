import * as ort from "onnxruntime-web/webgpu";
import "./style.css";

const MODEL_URL = "/models/yolo26n_webgpu.onnx";

const MODEL_NAME = "YOLO26n";
const MODEL_VARIANT = "NMS-free / one-to-one";

const INPUT_SIZE = 640;

/*
 * Display threshold only.
 * This does not change the trained model.
 */
const UI_CONFIDENCE = 0.45;

/*
 * Temporal confirmation for live webcam detections.
 */
const CONFIRM_FRAMES = 2;
const TRACK_TTL_MS = 650;
const TRACK_IOU = 0.30;

/*
 * Excluded from steady-state measurements.
 */
const WARMUP_RUNS = 6;

/*
 * Controlled browser benchmark.
 */
const BROWSER_BENCHMARK_WARMUPS = 10;
const BROWSER_BENCHMARK_RUNS = 60;

/*
 * COCO class names for the YOLO26n checkpoint.
 */
const COCO_NAMES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
];

const app = document.querySelector("#app");

app.innerHTML = `
  <main class="app-shell">

    <header class="header">
      <div>
        <p class="kicker">FOUNDRY / M3</p>

        <h1 class="title">
          RunsOnAnything
        </h1>

        <p class="subtitle">
          Local computer vision inference in the browser.
          WebGPU first, explicit WASM fallback, measured before optimization.
        </p>
      </div>

      <div id="status" class="status">
        <span class="status-dot"></span>
        <span id="statusText">Loading runtime</span>
      </div>
    </header>

    <section class="card">

      <div id="viewer" class="viewer">

        <video
          id="video"
          autoplay
          muted
          playsinline
        ></video>

        <img
          id="stillImage"
          alt="Local test frame"
        />

        <canvas id="overlay"></canvas>

        <div id="empty" class="viewer-empty">
          <div class="viewer-empty-inner">

            <p class="viewer-empty-title">
              Camera stopped
            </p>

            <p class="viewer-empty-text">
              Start the camera for live inference,
              or select a still image for an exact local test.
            </p>

          </div>
        </div>

      </div>

      <div class="controls">

        <div class="control-left">

          <button
            id="startBtn"
            class="btn btn-primary"
          >
            Start Camera
          </button>

          <button
            id="stopBtn"
            class="btn"
            disabled
          >
            Stop
          </button>

          <button
            id="benchBtn"
            class="btn"
            disabled
          >
            Benchmark 60 Runs
          </button>

          <label
            class="file-label"
            for="imageInput"
          >
            Test Image
          </label>

          <input
            id="imageInput"
            class="file-input"
            type="file"
            accept="image/png,image/jpeg,image/webp"
          />

        </div>

        <div class="control-right">

          <span class="chip">
            UI threshold ${Math.round(UI_CONFIDENCE * 100)}%
          </span>

          <span class="chip">
            ${CONFIRM_FRAMES}-frame confirmation
          </span>

        </div>

      </div>

      <div class="metrics">

        <div class="metric">
          <p class="metric-label">Runtime</p>
          <p id="runtime" class="metric-value">-</p>
          <p class="metric-sub">execution provider</p>
        </div>

        <div class="metric">
          <p class="metric-label">FPS</p>
          <p id="fps" class="metric-value">-</p>
          <p class="metric-sub">completed inferences / s</p>
        </div>

        <div class="metric">
          <p class="metric-label">Inference</p>
          <p id="inference" class="metric-value">-</p>
          <p class="metric-sub">rolling median</p>
        </div>

        <div class="metric">
          <p class="metric-label">Detections</p>
          <p id="detections" class="metric-value">0</p>
          <p class="metric-sub">confirmed boxes</p>
        </div>

        <div class="metric">
          <p class="metric-label">Model</p>
          <p id="modelSize" class="metric-value">-</p>
          <p class="metric-sub">download size</p>
        </div>

        <div class="metric">
          <p class="metric-label">Frame</p>
          <p id="frameTime" class="metric-value">-</p>
          <p class="metric-sub">end-to-end</p>
        </div>

      </div>

      <div class="tech">

        <span class="chip">
          MODEL <strong>${MODEL_NAME}</strong>
        </span>

        <span class="chip">
          VARIANT <strong>${MODEL_VARIANT}</strong>
        </span>

        <span class="chip">
          INPUT <strong>${INPUT_SIZE} x ${INPUT_SIZE}</strong>
        </span>

        <span class="chip">
          FORMAT <strong>ONNX</strong>
        </span>

        <span class="chip">
          NMS <strong>none in browser graph</strong>
        </span>

        <span class="chip">
          INPUT BUFFERS <strong>fresh per run</strong>
        </span>

        <span class="local-note">
          <span class="local-dot"></span>
          local inference - no frame-upload API
        </span>

      </div>

    </section>

    <section class="diagnostics">

      <div class="diag-card">

        <p class="diag-title">
          WebGPU / Runtime diagnostics
        </p>

        <p
          id="diagRuntime"
          class="diag-value"
        >
          -
        </p>

      </div>

      <div class="diag-card">

        <p class="diag-title">
          Last benchmark
        </p>

        <p
          id="diagBench"
          class="diag-value"
        >
          No benchmark run yet.
        </p>

      </div>

    </section>

    <footer class="footer">

      <span>
        FOUNDRY - Production Computer Vision Engineering
      </span>

      <span>
        BUILD -&gt; MEASURE -&gt; BREAK -&gt; OPTIMIZE -&gt; DEPLOY
      </span>

    </footer>

  </main>
`;

const video =
  document.querySelector("#video");

const overlay =
  document.querySelector("#overlay");

const overlayCtx =
  overlay.getContext("2d");

const stillImage =
  document.querySelector("#stillImage");

/*
 * CPU-side staging canvas.
 *
 * IMPORTANT:
 * The Float32Array below is NEVER passed directly
 * to ONNX Runtime WebGPU.
 *
 * It is only a staging buffer.
 */
const inputCanvas =
  document.createElement("canvas");

inputCanvas.width =
  INPUT_SIZE;

inputCanvas.height =
  INPUT_SIZE;

const inputCtx =
  inputCanvas.getContext(
    "2d",
    {
      willReadFrequently: true
    }
  );

const stagingData =
  new Float32Array(
    3 *
      INPUT_SIZE *
      INPUT_SIZE
  );

const statusEl =
  document.querySelector(
    "#status"
  );

const statusTextEl =
  document.querySelector(
    "#statusText"
  );

const runtimeEl =
  document.querySelector(
    "#runtime"
  );

const fpsEl =
  document.querySelector(
    "#fps"
  );

const inferenceEl =
  document.querySelector(
    "#inference"
  );

const detectionsEl =
  document.querySelector(
    "#detections"
  );

const modelSizeEl =
  document.querySelector(
    "#modelSize"
  );

const frameTimeEl =
  document.querySelector(
    "#frameTime"
  );

const diagRuntimeEl =
  document.querySelector(
    "#diagRuntime"
  );

const diagBenchEl =
  document.querySelector(
    "#diagBench"
  );

const emptyEl =
  document.querySelector(
    "#empty"
  );

const startBtn =
  document.querySelector(
    "#startBtn"
  );

const stopBtn =
  document.querySelector(
    "#stopBtn"
  );

const benchBtn =
  document.querySelector(
    "#benchBtn"
  );

const imageInput =
  document.querySelector(
    "#imageInput"
  );

let session =
  null;

let backend =
  "-";

let stream =
  null;

let running =
  false;

let inferenceBusy =
  false;

let rafId =
  null;

let currentTransform = {
  scale: 1,
  padX: 0,
  padY: 0
};

let tracks =
  [];

let nextTrackId =
  1;

let inferenceTimes =
  [];

let completedTimes =
  [];

let lastBenchmarkTensorData =
  null;

let stillObjectUrl =
  null;

/*
 * Every inference receives a brand-new
 * ArrayBuffer.
 *
 * This is the core fix for the detached-buffer
 * error.
 */
function createFreshTensor(
  data
) {
  const freshData =
    new Float32Array(
      data
    );

  return new ort.Tensor(
    "float32",
    freshData,
    [
      1,
      3,
      INPUT_SIZE,
      INPUT_SIZE
    ]
  );
}

function setStatus(
  message,
  state = ""
) {
  statusTextEl.textContent =
    message;

  statusEl.className =
    `status ${state}`;
}

function fmtMs(
  value
) {
  return Number.isFinite(
    value
  )
    ? `${value.toFixed(1)} ms`
    : "-";
}

function median(
  values
) {
  if (
    !values.length
  ) {
    return NaN;
  }

  const sorted =
    [
      ...values
    ].sort(
      (a, b) => a - b
    );

  const mid =
    Math.floor(
      sorted.length / 2
    );

  return (
    sorted.length % 2
      ? sorted[mid]
      : (
          sorted[mid - 1] +
          sorted[mid]
        ) / 2
  );
}

function percentile(
  values,
  p
) {
  if (
    !values.length
  ) {
    return NaN;
  }

  const sorted =
    [
      ...values
    ].sort(
      (a, b) => a - b
    );

  const index =
    (
      sorted.length - 1
    ) * p;

  const low =
    Math.floor(
      index
    );

  const high =
    Math.ceil(
      index
    );

  if (
    low === high
  ) {
    return sorted[low];
  }

  return (
    sorted[low] +
    (
      sorted[high] -
      sorted[low]
    ) *
      (
        index - low
      )
  );
}

function iou(
  a,
  b
) {
  const x1 =
    Math.max(
      a.x1,
      b.x1
    );

  const y1 =
    Math.max(
      a.y1,
      b.y1
    );

  const x2 =
    Math.min(
      a.x2,
      b.x2
    );

  const y2 =
    Math.min(
      a.y2,
      b.y2
    );

  const width =
    Math.max(
      0,
      x2 - x1
    );

  const height =
    Math.max(
      0,
      y2 - y1
    );

  const intersection =
    width * height;

  const areaA =
    Math.max(
      0,
      a.x2 - a.x1
    ) *
    Math.max(
      0,
      a.y2 - a.y1
    );

  const areaB =
    Math.max(
      0,
      b.x2 - b.x1
    ) *
    Math.max(
      0,
      b.y2 - b.y1
    );

  const union =
    areaA +
    areaB -
    intersection;

  return union > 0
    ? intersection / union
    : 0;
}

async function hasWebGPU() {
  if (
    !navigator.gpu
  ) {
    return false;
  }

  try {
    const adapter =
      await navigator.gpu.requestAdapter(
        {
          powerPreference:
            "high-performance"
        }
      );

    return Boolean(
      adapter
    );

  } catch {
    return false;
  }
}

async function describeWebGPU() {
  if (
    !navigator.gpu
  ) {
    return (
      "navigator.gpu: unavailable"
    );
  }

  try {
    const adapter =
      await navigator.gpu.requestAdapter(
        {
          powerPreference:
            "high-performance"
        }
      );

    if (!adapter) {
      return (
        "navigator.gpu: present | " +
        "adapter: unavailable"
      );
    }

    const info =
      adapter.info ?? {};

    return [
      "backend: WebGPU",
      `vendor: ${
        info.vendor ||
        "unknown"
      }`,
      `architecture: ${
        info.architecture ||
        "unknown"
      }`,
      `device: ${
        info.device ||
        "unknown"
      }`,
      `description: ${
        info.description ||
        "unknown"
      }`
    ].join(
      " | "
    );

  } catch (error) {
    return (
      "WebGPU diagnostics failed: " +
      (
        error?.message ||
        error
      )
    );
  }
}

async function fetchModelSize() {
  try {
    const response =
      await fetch(
        MODEL_URL,
        {
          method: "HEAD",
          cache: "no-store"
        }
      );

    const bytes =
      Number(
        response.headers.get(
          "content-length"
        )
      );

    if (
      Number.isFinite(
        bytes
      ) &&
      bytes > 0
    ) {
      modelSizeEl.textContent =
        `${(
          bytes /
          (
            1024 *
            1024
          )
        ).toFixed(1)} MB`;

      return;
    }

  } catch {
    // Some development servers do not expose Content-Length.
  }

  modelSizeEl.textContent =
    "~9.4 MB";
}

async function createSession() {
  /*
   * We are using the WebGPU-specific package.
   *
   * WASM proxy is disabled because we do not want
   * its worker transport involved in the primary
   * WebGPU path.
   */
  ort.env.logLevel =
    "warning";

  ort.env.wasm.proxy =
    false;

  const webgpuAvailable =
    await hasWebGPU();

  const runtimeInfo =
    await describeWebGPU();

  diagRuntimeEl.textContent =
    runtimeInfo;

  if (
    webgpuAvailable
  ) {
    try {
      setStatus(
        "Loading WebGPU model"
      );

      session =
        await ort.InferenceSession.create(
          MODEL_URL,
          {
            executionProviders: [
              {
                name: "webgpu",

                preferredLayout:
                  "NCHW",

                storageBufferCacheMode:
                  "simple",

                validationMode:
                  "basic"
              }
            ],

            graphOptimizationLevel:
              "all",

            /*
             * Keep graph capture disabled
             * until the basic WebGPU path
             * is verified.
             */
            enableGraphCapture:
              false
          }
        );

      backend =
        "WebGPU";

      runtimeEl.textContent =
        backend;

      diagRuntimeEl.textContent +=
        "\nmodel: " +
        MODEL_URL +
        "\ninput names: " +
        JSON.stringify(
          session.inputNames
        ) +
        "\noutput names: " +
        JSON.stringify(
          session.outputNames
        );

      setStatus(
        "Ready - WebGPU"
      );

      return;

    } catch (
      webgpuError
    ) {
      console.warn(
        "WebGPU session creation failed. " +
        "Falling back to WASM.",
        webgpuError
      );

      diagRuntimeEl.textContent +=
        "\nWebGPU session error: " +
        (
          webgpuError?.message ||
          webgpuError
        );
    }
  }

  /*
   * Explicit CPU/WASM fallback.
   *
   * This is used only if the WebGPU session
   * cannot be created.
   */
  setStatus(
    "Loading WASM fallback"
  );

  session =
    await ort.InferenceSession.create(
      MODEL_URL,
      {
        executionProviders: [
          "wasm"
        ],

        graphOptimizationLevel:
          "all"
      }
    );

  backend =
    "WASM";

  runtimeEl.textContent =
    backend;

  setStatus(
    "Ready - WASM",
    "warn"
  );

  diagRuntimeEl.textContent +=
    "\nmodel: " +
    MODEL_URL +
    "\ninput names: " +
    JSON.stringify(
      session.inputNames
    ) +
    "\noutput names: " +
    JSON.stringify(
      session.outputNames
    );
}

function resizeOverlay() {
  if (
    !video.videoWidth ||
    !video.videoHeight
  ) {
    return;
  }

  overlay.width =
    video.videoWidth;

  overlay.height =
    video.videoHeight;
}

/*
 * Preprocess into the CPU staging buffer.
 *
 * Returns the staging array.
 *
 * IMPORTANT:
 * The returned array is copied before being
 * passed to ORT.
 */
function preprocessSource(
  source,
  sourceWidth,
  sourceHeight
) {
  const scale =
    Math.min(
      INPUT_SIZE /
        sourceWidth,
      INPUT_SIZE /
        sourceHeight
    );

  const drawWidth =
    Math.round(
      sourceWidth *
        scale
    );

  const drawHeight =
    Math.round(
      sourceHeight *
        scale
    );

  const padX =
    Math.floor(
      (
        INPUT_SIZE -
        drawWidth
      ) / 2
    );

  const padY =
    Math.floor(
      (
        INPUT_SIZE -
        drawHeight
      ) / 2
    );

  inputCtx.fillStyle =
    "rgb(114,114,114)";

  inputCtx.fillRect(
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE
  );

  inputCtx.drawImage(
    source,
    0,
    0,
    sourceWidth,
    sourceHeight,
    padX,
    padY,
    drawWidth,
    drawHeight
  );

  const rgba =
    inputCtx.getImageData(
      0,
      0,
      INPUT_SIZE,
      INPUT_SIZE
    ).data;

  const plane =
    INPUT_SIZE *
    INPUT_SIZE;

  for (
    let i = 0;
    i < plane;
    i += 1
  ) {
    stagingData[i] =
      rgba[i * 4] /
      255;

    stagingData[
      plane + i
    ] =
      rgba[
        i * 4 + 1
      ] / 255;

    stagingData[
      2 * plane + i
    ] =
      rgba[
        i * 4 + 2
      ] / 255;
  }

  currentTransform = {
    scale,
    padX,
    padY
  };

  return stagingData;
}

function preprocessVideoFrame() {
  if (
    !video.videoWidth ||
    !video.videoHeight
  ) {
    return null;
  }

  return preprocessSource(
    video,
    video.videoWidth,
    video.videoHeight
  );
}

function preprocessImage(
  image
) {
  return preprocessSource(
    image,
    image.naturalWidth ||
      image.width,
    image.naturalHeight ||
      image.height
  );
}

function decodeOutput(
  results
) {
  const outputName =
    session.outputNames[0];

  const tensor =
    results[outputName];

  if (
    !tensor?.data ||
    !tensor?.dims
  ) {
    return [];
  }

  const dims =
    tensor.dims;

  const data =
    tensor.data;

  const ROW_SIZE =
    6;

  let rows =
    0;

  if (
    dims.length === 3 &&
    dims[2] === ROW_SIZE
  ) {
    rows =
      dims[1];

  } else if (
    dims.length === 2 &&
    dims[1] === ROW_SIZE
  ) {
    rows =
      dims[0];

  } else {
    throw new Error(
      `Unexpected output shape: ${JSON.stringify(
        dims
      )}`
    );
  }

  const detections =
    [];

  for (
    let i = 0;
    i < rows;
    i += 1
  ) {
    const base =
      i * ROW_SIZE;

    const x1 =
      Number(
        data[base]
      );

    const y1 =
      Number(
        data[
          base + 1
        ]
      );

    const x2 =
      Number(
        data[
          base + 2
        ]
      );

    const y2 =
      Number(
        data[
          base + 3
        ]
      );

    const confidence =
      Number(
        data[
          base + 4
        ]
      );

    const classId =
      Math.round(
        Number(
          data[
            base + 5
          ]
        )
      );

    if (
      !Number.isFinite(
        confidence
      )
    ) {
      continue;
    }

    if (
      confidence <
      UI_CONFIDENCE
    ) {
      continue;
    }

    if (
      !Number.isFinite(
        classId
      ) ||
      classId < 0 ||
      classId >=
        COCO_NAMES.length
    ) {
      continue;
    }

    if (
      !(x2 > x1) ||
      !(y2 > y1)
    ) {
      continue;
    }

    detections.push({
      classId,

      className:
        COCO_NAMES[
          classId
        ],

      confidence,

      x1:
        Math.max(
          0,
          Math.min(
            INPUT_SIZE,
            x1
          )
        ),

      y1:
        Math.max(
          0,
          Math.min(
            INPUT_SIZE,
            y1
          )
        ),

      x2:
        Math.max(
          0,
          Math.min(
            INPUT_SIZE,
            x2
          )
        ),

      y2:
        Math.max(
          0,
          Math.min(
            INPUT_SIZE,
            y2
          )
        )
    });
  }

  return detections;
}

function toVideoBox(
  det
) {
  const {
    scale,
    padX,
    padY
  } =
    currentTransform;

  return {
    ...det,

    x1:
      (
        det.x1 -
        padX
      ) / scale,

    y1:
      (
        det.y1 -
        padY
      ) / scale,

    x2:
      (
        det.x2 -
        padX
      ) / scale,

    y2:
      (
        det.y2 -
        padY
      ) / scale
  };
}

function stabilize(
  detections,
  now
) {
  const nextTracks =
    [];

  const usedTrackIds =
    new Set();

  for (
    const det of detections
  ) {
    let best =
      null;

    let bestIou =
      0;

    for (
      const track of tracks
    ) {
      if (
        usedTrackIds.has(
          track.id
        )
      ) {
        continue;
      }

      if (
        track.classId !==
        det.classId
      ) {
        continue;
      }

      const score =
        iou(
          track.box,
          det
        );

      if (
        score >
          bestIou &&
        score >=
          TRACK_IOU
      ) {
        best =
          track;

        bestIou =
          score;
      }
    }

    if (best) {
      usedTrackIds.add(
        best.id
      );

      nextTracks.push({
        ...best,

        box: det,

        hits:
          Math.min(
            CONFIRM_FRAMES + 1,
            best.hits + 1
          ),

        lastSeen:
          now,

        confidence:
          det.confidence
      });

    } else {
      nextTracks.push({
        id:
          nextTrackId++,

        classId:
          det.classId,

        box: det,

        hits: 1,

        lastSeen:
          now,

        confidence:
          det.confidence
      });
    }
  }

  for (
    const old of tracks
  ) {
    if (
      usedTrackIds.has(
        old.id
      )
    ) {
      continue;
    }

    if (
      now -
        old.lastSeen <=
      TRACK_TTL_MS
    ) {
      nextTracks.push(
        old
      );
    }
  }

  tracks =
    nextTracks;

  return tracks
    .filter(
      (
        track
      ) =>
        track.hits >=
          CONFIRM_FRAMES &&
        now -
          track.lastSeen <=
          TRACK_TTL_MS
    )
    .map(
      (
        track
      ) =>
        toVideoBox(
          track.box
        )
    );
}

function drawDetections(
  detections
) {
  if (
    !overlay.width ||
    !overlay.height
  ) {
    return;
  }

  overlayCtx.clearRect(
    0,
    0,
    overlay.width,
    overlay.height
  );

  overlayCtx.lineWidth =
    Math.max(
      2,
      overlay.width / 600
    );

  overlayCtx.font =
    `${Math.max(
      12,
      overlay.width / 95
    )}px Inter, sans-serif`;

  for (
    const det of detections
  ) {
    const x1 =
      Math.max(
        0,
        Math.min(
          overlay.width,
          det.x1
        )
      );

    const y1 =
      Math.max(
        0,
        Math.min(
          overlay.height,
          det.y1
        )
      );

    const x2 =
      Math.max(
        0,
        Math.min(
          overlay.width,
          det.x2
        )
      );

    const y2 =
      Math.max(
        0,
        Math.min(
          overlay.height,
          det.y2
        )
      );

    const label =
      `${det.className} ${
        (
          det.confidence *
          100
        ).toFixed(0)
      }%`;

    const textWidth =
      overlayCtx.measureText(
        label
      ).width;

    const boxHeight =
      24;

    overlayCtx.strokeStyle =
      "#ffffff";

    overlayCtx.strokeRect(
      x1,
      y1,
      x2 - x1,
      y2 - y1
    );

    overlayCtx.fillStyle =
      "rgba(0,0,0,.78)";

    overlayCtx.fillRect(
      x1,
      Math.max(
        0,
        y1 - boxHeight
      ),
      textWidth + 12,
      boxHeight
    );

    overlayCtx.fillStyle =
      "#ffffff";

    overlayCtx.fillText(
      label,
      x1 + 6,
      Math.max(
        17,
        y1 - 7
      )
    );
  }
}

function updateFps(
  now
) {
  completedTimes.push(
    now
  );

  const cutoff =
    now - 1200;

  while (
    completedTimes.length &&
    completedTimes[0] <
      cutoff
  ) {
    completedTimes.shift();
  }

  if (
    completedTimes.length >=
    2
  ) {
    const span =
      (
        completedTimes[
          completedTimes.length - 1
        ] -
        completedTimes[0]
      ) / 1000;

    const fps =
      span > 0
        ? (
            completedTimes.length -
            1
          ) / span
        : NaN;

    fpsEl.textContent =
      Number.isFinite(
        fps
      )
        ? fps.toFixed(2)
        : "-";
  }
}

/*
 * Runs one live inference.
 *
 * CORE FIX:
 * preprocess -> stagingData
 * stagingData -> NEW Float32Array
 * NEW Float32Array -> NEW Tensor
 * NEW Tensor -> session.run()
 *
 * The input buffer is therefore never reused after
 * ONNX Runtime receives it.
 */
async function runOneInference() {
  if (!session) {
    throw new Error(
      "Inference session is not ready."
    );
  }

  const frameStart =
    performance.now();

  const staged =
    preprocessVideoFrame();

  if (!staged) {
    return;
  }

  /*
   * IMPORTANT:
   * Create a fresh tensor and fresh ArrayBuffer.
   */
  const tensor =
    createFreshTensor(
      staged
    );

  const inferenceStart =
    performance.now();

  const results =
    await session.run({
      [session.inputNames[0]]:
        tensor
    });

  const inferenceMs =
    performance.now() -
    inferenceStart;

  const totalMs =
    performance.now() -
    frameStart;

  const raw =
    decodeOutput(
      results
    );

  const now =
    performance.now();

  const confirmed =
    stabilize(
      raw,
      now
    );

  inferenceTimes.push(
    inferenceMs
  );

  if (
    inferenceTimes.length >
    21
  ) {
    inferenceTimes.shift();
  }

  inferenceEl.textContent =
    fmtMs(
      median(
        inferenceTimes
      )
    );

  frameTimeEl.textContent =
    fmtMs(
      totalMs
    );

  detectionsEl.textContent =
    String(
      confirmed.length
    );

  updateFps(
    now
  );

  drawDetections(
    confirmed
  );

  /*
   * Save a CPU copy for the fixed-input
   * benchmark.
   *
   * This copy is never transferred directly
   * to ORT.
   */
  lastBenchmarkTensorData =
    new Float32Array(
      staged
    );
}

async function warmupFromCurrentFrame() {
  const staged =
    preprocessVideoFrame();

  if (!staged) {
    throw new Error(
      "Camera frame is not ready."
    );
  }

  setStatus(
    `Warming up ${backend}`
  );

  for (
    let i = 0;
    i < WARMUP_RUNS;
    i += 1
  ) {
    /*
     * Fresh Tensor for every warmup run.
     */
    const tensor =
      createFreshTensor(
        staged
      );

    await session.run({
      [session.inputNames[0]]:
        tensor
    });
  }

  inferenceTimes =
    [];

  completedTimes =
    [];

  tracks =
    [];
}

async function startCamera() {
  if (!session) {
    await createSession();
  }

  if (stream) {
    return;
  }

  try {
    if (stillObjectUrl) {
      URL.revokeObjectURL(
        stillObjectUrl
      );

      stillObjectUrl =
        null;
    }

    stillImage.classList.remove(
      "visible"
    );

    stillImage.removeAttribute(
      "src"
    );

    video.style.visibility =
      "visible";

    stream =
      await navigator.mediaDevices.getUserMedia(
        {
          video: {
            facingMode:
              "user",

            width: {
              ideal: 1280
            },

            height: {
              ideal: 720
            }
          },

          audio: false
        }
      );

    video.srcObject =
      stream;

    await video.play();

    resizeOverlay();

    emptyEl.style.display =
      "none";

    startBtn.disabled =
      true;

    stopBtn.disabled =
      false;

    benchBtn.disabled =
      false;

    await warmupFromCurrentFrame();

    running =
      true;

    setStatus(
      `Running - ${backend}`,
      "running"
    );

    renderLoop();

  } catch (error) {
    stopCamera();

    setStatus(
      "Camera error",
      "warn"
    );

    console.error(
      error
    );

    alert(
      `Camera/inference startup failed:\n\n${
        error?.message ||
        error
      }`
    );
  }
}

function stopCamera() {
  running =
    false;

  inferenceBusy =
    false;

  if (rafId) {
    cancelAnimationFrame(
      rafId
    );
  }

  rafId =
    null;

  tracks =
    [];

  drawDetections(
    []
  );

  if (stream) {
    for (
      const track of
        stream.getTracks()
    ) {
      track.stop();
    }
  }

  stream =
    null;

  video.srcObject =
    null;

  stillImage.classList.remove(
    "visible"
  );

  stillImage.removeAttribute(
    "src"
  );

  if (stillObjectUrl) {
    URL.revokeObjectURL(
      stillObjectUrl
    );

    stillObjectUrl =
      null;
  }

  video.style.visibility =
    "visible";

  emptyEl.style.display =
    "grid";

  startBtn.disabled =
    false;

  stopBtn.disabled =
    true;

  benchBtn.disabled =
    !session;

  setStatus(
    session
      ? `Ready - ${backend}`
      : "Runtime not loaded"
  );
}

function renderLoop() {
  if (!running) {
    return;
  }

  rafId =
    requestAnimationFrame(
      renderLoop
    );

  if (
    inferenceBusy ||
    video.readyState <
      HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    return;
  }

  inferenceBusy =
    true;

  runOneInference()
    .catch(
      (error) => {
        running =
          false;

        setStatus(
          "Inference error",
          "warn"
        );

        console.error(
          error
        );

        diagRuntimeEl.textContent +=
          "\nRuntime inference error: " +
          (
            error?.message ||
            error
          );
      }
    )
    .finally(
      () => {
        inferenceBusy =
          false;
      }
    );
}

/*
 * Controlled benchmark.
 *
 * We measure session.run() only.
 *
 * Each iteration creates a fresh Tensor
 * from a fresh ArrayBuffer so the worker can
 * transfer/detach it safely.
 */
async function runBrowserBenchmark() {
  if (!session) {
    await createSession();
  }

  if (
    !lastBenchmarkTensorData
  ) {
    const staged =
      preprocessVideoFrame();

    if (!staged) {
      throw new Error(
        "Start the camera or provide a test image first."
      );
    }

    lastBenchmarkTensorData =
      new Float32Array(
        staged
      );
  }

  const wasRunning =
    running;

  running =
    false;

  if (rafId) {
    cancelAnimationFrame(
      rafId
    );

    rafId =
      null;
  }

  try {
    setStatus(
      `Benchmarking ${backend}`
    );

    /*
     * Warmup.
     */
    for (
      let i = 0;
      i <
        BROWSER_BENCHMARK_WARMUPS;
      i += 1
    ) {
      const tensor =
        createFreshTensor(
          lastBenchmarkTensorData
        );

      await session.run({
        [session.inputNames[0]]:
          tensor
      });
    }

    const timings =
      [];

    /*
     * Measured runs.
     */
    for (
      let i = 0;
      i <
        BROWSER_BENCHMARK_RUNS;
      i += 1
    ) {
      /*
       * Fresh tensor for EVERY run.
       *
       * Created before the stopwatch so the
       * reported number remains the model/runtime
       * session.run timing.
       */
      const tensor =
        createFreshTensor(
          lastBenchmarkTensorData
        );

      const start =
        performance.now();

      await session.run({
        [session.inputNames[0]]:
          tensor
      });

      timings.push(
        performance.now() -
        start
      );
    }

    const mean =
      timings.reduce(
        (
          sum,
          value
        ) =>
          sum + value,
        0
      ) /
      timings.length;

    const p50 =
      median(
        timings
      );

    const p95 =
      percentile(
        timings,
        0.95
      );

    const fps =
      1000 / p50;

    diagBenchEl.textContent =
      `backend: ${backend}\n` +
      `runs: ${BROWSER_BENCHMARK_RUNS}\n` +
      `input: ${INPUT_SIZE}x${INPUT_SIZE}\n` +
      `mean: ${mean.toFixed(2)} ms\n` +
      `p50: ${p50.toFixed(2)} ms\n` +
      `p95: ${p95.toFixed(2)} ms\n` +
      `equivalent throughput: ${fps.toFixed(2)} FPS`;

    inferenceEl.textContent =
      fmtMs(
        p50
      );

    fpsEl.textContent =
      fps.toFixed(2);

  } finally {
    if (wasRunning) {
      running =
        true;

      setStatus(
        `Running - ${backend}`,
        "running"
      );

      renderLoop();

    } else {
      setStatus(
        `Ready - ${backend}`
      );
    }
  }
}

async function runTestImage(
  file
) {
  if (!session) {
    await createSession();
  }

  const objectUrl =
    URL.createObjectURL(
      file
    );

  const image =
    new Image();

  image.decoding =
    "async";

  image.src =
    objectUrl;

  await image.decode();

  try {
    const staged =
      preprocessImage(
        image
      );

    /*
     * Create a fresh Tensor for each
     * inference of the still image.
     */
    setStatus(
      `Testing image - ${backend}`
    );

    for (
      let i = 0;
      i < 3;
      i += 1
    ) {
      const tensor =
        createFreshTensor(
          staged
        );

      await session.run({
        [session.inputNames[0]]:
          tensor
      });
    }

    const tensor =
      createFreshTensor(
        staged
      );

    const start =
      performance.now();

    const results =
      await session.run({
        [session.inputNames[0]]:
          tensor
      });

    const inferenceMs =
      performance.now() -
      start;

    const raw =
      decodeOutput(
        results
      );

    /*
     * Still images do not use temporal
     * confirmation.
     */
    const detections =
      raw.map(
        toVideoBox
      );

    if (stream) {
      for (
        const track of
          stream.getTracks()
      ) {
        track.stop();
      }
    }

    stream =
      null;

    running =
      false;

    if (rafId) {
      cancelAnimationFrame(
        rafId
      );

      rafId =
        null;
    }

    video.srcObject =
      null;

    video.style.visibility =
      "hidden";

    stillObjectUrl =
      objectUrl;

    stillImage.src =
      stillObjectUrl;

    stillImage.classList.add(
      "visible"
    );

    emptyEl.style.display =
      "none";

    startBtn.disabled =
      false;

    stopBtn.disabled =
      true;

    benchBtn.disabled =
      false;

    /*
     * Keep a CPU-side copy for benchmark.
     */
    lastBenchmarkTensorData =
      new Float32Array(
        staged
      );

    inferenceEl.textContent =
      fmtMs(
        inferenceMs
      );

    frameTimeEl.textContent =
      fmtMs(
        inferenceMs
      );

    detectionsEl.textContent =
      String(
        detections.length
      );

    fpsEl.textContent =
      "-";

    overlay.width =
      image.naturalWidth;

    overlay.height =
      image.naturalHeight;

    drawDetections(
      detections
    );

    setStatus(
      `Test image - ${backend}`
    );

  } catch (error) {
    URL.revokeObjectURL(
      objectUrl
    );

    throw error;
  }
}

startBtn.addEventListener(
  "click",
  () => {
    startCamera();
  }
);

stopBtn.addEventListener(
  "click",
  () => {
    stopCamera();
  }
);

benchBtn.addEventListener(
  "click",
  () => {
    runBrowserBenchmark()
      .catch(
        (error) => {
          setStatus(
            "Benchmark error",
            "warn"
          );

          console.error(
            error
          );

          diagBenchEl.textContent =
            String(
              error?.stack ||
              error
            );

          alert(
            `Browser benchmark failed:\n\n${
              error?.message ||
              error
            }`
          );
        }
      );
  }
);

imageInput.addEventListener(
  "change",
  (event) => {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    runTestImage(
      file
    ).catch(
      (error) => {
        setStatus(
          "Image test error",
          "warn"
        );

        console.error(
          error
        );

        alert(
          `Test image failed:\n\n${
            error?.message ||
            error
          }`
        );
      }
    );
  }
);

video.addEventListener(
  "loadedmetadata",
  resizeOverlay
);

window.addEventListener(
  "resize",
  resizeOverlay
);

document.addEventListener(
  "visibilitychange",
  () => {
    if (
      document.hidden &&
      stream
    ) {
      stopCamera();
    }
  }
);

/*
 * Startup.
 */
(async () => {
  try {
    await fetchModelSize();

    await createSession();

  } catch (error) {
    setStatus(
      "Runtime error",
      "warn"
    );

    diagRuntimeEl.textContent =
      String(
        error?.stack ||
        error
      );

    console.error(
      error
    );
  }
})();