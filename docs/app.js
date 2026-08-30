const pixelDefaults = [10, 10, 10, 10, 50, 10, 10, 10, 10];
const kernelPresets = {
  sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
  blur: Array(9).fill(1 / 9),
  edge: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
};

const pixelMatrix = document.querySelector("#pixel-matrix");
const kernelMatrix = document.querySelector("#kernel-matrix");

function buildMatrix(container, values, kind) {
  container.replaceChildren();
  values.forEach((value, index) => {
    const input = document.createElement("input");
    input.type = "number";
    input.step = kind === "kernel" ? "0.01" : "1";
    input.value = Number.isInteger(value) ? String(value) : value.toFixed(3);
    input.dataset.kind = kind;
    input.dataset.index = String(index);
    input.setAttribute("aria-label", `${kind === "pixel" ? "像素" : "卷积核"} ${index + 1}`);
    if (index === 4) input.classList.add("center");
    input.addEventListener("input", calculateConvolution);
    container.append(input);
  });
}

function matrixValues(container) {
  return [...container.querySelectorAll("input")].map((input) => Number(input.value) || 0);
}

function calculateConvolution() {
  const pixels = matrixValues(pixelMatrix);
  const kernel = matrixValues(kernelMatrix);
  const flipped = [...kernel].reverse();
  const products = pixels.map((value, index) => value * flipped[index]);
  const result = products.reduce((sum, value) => sum + value, 0);
  document.querySelector("#convolution-result").textContent = Number(result.toFixed(3)).toString();
  document.querySelector("#convolution-equation").textContent = products
    .filter((value) => Math.abs(value) > 1e-9)
    .map((value) => Number(value.toFixed(2)).toString())
    .join(" + ") || "所有乘积均为 0";
}

buildMatrix(pixelMatrix, pixelDefaults, "pixel");
buildMatrix(kernelMatrix, kernelPresets.sharpen, "kernel");
calculateConvolution();

const matrixScroller = document.querySelector(".matrix-scroll");
let dragStartX = 0;
let dragStartScroll = 0;
let pendingDragScroll = 0;
let dragFrame = 0;

matrixScroller.addEventListener("pointerdown", (event) => {
  if (event.pointerType !== "mouse" || event.target.closest("input, button")) return;
  dragStartX = event.clientX;
  dragStartScroll = matrixScroller.scrollLeft;
  pendingDragScroll = dragStartScroll;
  matrixScroller.classList.add("dragging");
  matrixScroller.setPointerCapture(event.pointerId);
});

matrixScroller.addEventListener("pointermove", (event) => {
  if (!matrixScroller.classList.contains("dragging")) return;
  pendingDragScroll = dragStartScroll - (event.clientX - dragStartX);
  if (dragFrame) return;
  dragFrame = requestAnimationFrame(() => {
    matrixScroller.scrollLeft = pendingDragScroll;
    dragFrame = 0;
  });
});

function stopMatrixDrag(event) {
  if (!matrixScroller.classList.contains("dragging")) return;
  matrixScroller.classList.remove("dragging");
  if (matrixScroller.hasPointerCapture(event.pointerId)) matrixScroller.releasePointerCapture(event.pointerId);
}

matrixScroller.addEventListener("pointerup", stopMatrixDrag);
matrixScroller.addEventListener("pointercancel", stopMatrixDrag);

function scrollMatrix(direction) {
  matrixScroller.scrollBy({ left: direction * Math.min(230, matrixScroller.clientWidth * 0.72), behavior: "smooth" });
}

document.querySelectorAll("[data-scroll-matrix]").forEach((button) => {
  button.addEventListener("click", () => scrollMatrix(Number(button.dataset.scrollMatrix)));
});

matrixScroller.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    scrollMatrix(event.key === "ArrowLeft" ? -1 : 1);
  }
});

document.querySelectorAll("[data-kernel]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-kernel]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    buildMatrix(kernelMatrix, kernelPresets[button.dataset.kernel], "kernel");
    calculateConvolution();
  });
});

const cannyStages = [
  { x: 0, y: 0, label: "1 / 6 · 输入图像", description: "输入图同时包含直线、曲线、角点和噪声，是观察不同阶段的统一基准。" },
  { x: 1, y: 0, label: "2 / 6 · Gaussian 平滑", description: "先抑制随机高频噪声，避免它在求导后变成大量虚假边缘。" },
  { x: 2, y: 0, label: "3 / 6 · 梯度幅值", description: "Sobel 水平与垂直导数组合成幅值；亮处表示局部强度变化明显。" },
  { x: 0, y: 1, label: "4 / 6 · 梯度方向", description: "方向描述强度上升最快的方向，它与边缘走向互相垂直。" },
  { x: 1, y: 1, label: "5 / 6 · 非极大值抑制", description: "沿梯度方向只保留局部最大值，把宽梯度脊线压缩成细边缘。" },
  { x: 2, y: 1, label: "6 / 6 · 滞后连接结果", description: "强边缘作为种子，保留与之连通的弱候选，删除孤立噪声。" },
];

document.querySelectorAll("[data-stage]").forEach((button) => {
  button.addEventListener("click", () => {
    const stage = cannyStages[Number(button.dataset.stage)];
    document.querySelectorAll("[data-stage]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.querySelector("#canny-stage-image").style.transform = `translate(${-stage.x * 256}px, ${-stage.y * 286}px)`;
    document.querySelector("#canny-stage-label").textContent = stage.label;
    document.querySelector("#canny-description").textContent = stage.description;
  });
});

const regions = {
  flat: { name: "平坦区", max: 0.05, min: 0.03, summary: "小 / 小 → 平坦区域", explanation: "两个方向都缺少明显亮度变化，窗口移动后几乎不变。" },
  edge: { name: "边缘", max: 0.95, min: 0.06, summary: "大 / 小 → 边缘", explanation: "跨越边缘时变化强，沿着边缘移动时变化弱，因此只有一个大特征值。" },
  corner: { name: "角点", max: 0.96, min: 0.78, summary: "大 / 大 → 稳定角点", explanation: "两个主要方向都有强烈变化，窗口向多数方向移动都会改变外观。" },
};

function renderRegion(key) {
  const region = regions[key];
  document.querySelector("#region-classification").textContent = region.name;
  document.querySelector("#region-explanation").textContent = region.explanation;
  document.querySelector("#lambda-max-value").textContent = region.max.toFixed(2);
  document.querySelector("#lambda-min-value").textContent = region.min.toFixed(2);
  document.querySelector("#lambda-max-bar").style.width = `${region.max * 100}%`;
  document.querySelector("#lambda-min-bar").style.width = `${region.min * 100}%`;
  document.querySelector("#eigen-summary").textContent = region.summary;
}

document.querySelectorAll("[data-region]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-region]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderRegion(button.dataset.region);
  });
});
renderRegion("flat");

const reprojectionErrors = [0.18, 0.24, 0.31, 0.38, 0.44, 0.51, 0.62, 0.74, 0.88, 1.02, 1.19, 1.42, 1.76, 5.9, 7.4];
const thresholdSlider = document.querySelector("#ransac-threshold");

function renderRansacThreshold() {
  const threshold = Number(thresholdSlider.value);
  const inliers = reprojectionErrors.filter((error) => error <= threshold).length;
  const outliers = reprojectionErrors.length - inliers;
  document.querySelector("#ransac-threshold-value").textContent = `${threshold.toFixed(1)} px`;
  document.querySelector("#ransac-inlier-count").textContent = String(inliers);
  document.querySelector("#ransac-outlier-count").textContent = String(outliers);
  document.querySelector("#ransac-inlier-ratio").textContent = `${Math.round(100 * inliers / reprojectionErrors.length)}%`;
  document.querySelector("#ransac-guidance").textContent = threshold < 1.5
    ? "阈值偏严：可靠对应也可能被误删，模型可用点减少。"
    : threshold <= 3.0
      ? "阈值适中：保留真实对应，同时拒绝明显错误。"
      : "阈值偏松：错误对应开始混入内点，几何模型可能被拉偏。";
}

thresholdSlider.addEventListener("input", renderRansacThreshold);
renderRansacThreshold();

const focalSlider = document.querySelector("#focal-slider");
const baselineSlider = document.querySelector("#baseline-slider");
const disparitySlider = document.querySelector("#disparity-slider");

function renderStereoDepth() {
  const focal = Number(focalSlider.value);
  const baseline = Number(baselineSlider.value);
  const disparity = Number(disparitySlider.value);
  const depth = focal * baseline / disparity;
  const nearDepth = focal * baseline / (disparity + 0.5);
  const farDepth = focal * baseline / Math.max(disparity - 0.5, 0.5);
  const uncertainty = Math.max(depth - nearDepth, farDepth - depth);
  document.querySelector("#focal-value").textContent = `${focal} px`;
  document.querySelector("#baseline-value").textContent = `${baseline.toFixed(2)} m`;
  document.querySelector("#disparity-value").textContent = `${disparity} px`;
  document.querySelector("#depth-value").textContent = `${depth.toFixed(2)} m`;
  document.querySelector("#depth-formula").textContent = `${focal} × ${baseline.toFixed(2)} ÷ ${disparity}`;
  document.querySelector("#depth-uncertainty").textContent = `±0.5 px 匹配误差约产生 ±${uncertainty.toFixed(2)} m 深度变化`;
  document.querySelector("#uncertainty-bar").style.width = `${Math.min(100, 8 + uncertainty * 18)}%`;
}

[focalSlider, baselineSlider, disparitySlider].forEach((slider) => slider.addEventListener("input", renderStereoDepth));
renderStereoDepth();

const themeToggle = document.querySelector("#theme-toggle");
const storedTheme = localStorage.getItem("v2a-theme");
if (storedTheme) document.documentElement.dataset.theme = storedTheme;
themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("v2a-theme", next);
});


// ---------- Hough accumulator lab ----------

function houghScene(kind) {
  const width = 220;
  const height = 150;
  const edges = Array.from({ length: height }, () => new Uint8Array(width));
  const mark = (x, y) => {
    if (x >= 0 && x < width && y >= 0 && y < height) edges[y][x] = 1;
  };
  const line = (x0, y0, x1, y1, thickness) => {
    let dx = Math.abs(x1 - x0);
    let dy = -Math.abs(y1 - y0);
    let sx = x0 < x1 ? 1 : -1;
    let sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    const radius = Math.floor(thickness / 2);
    while (true) {
      for (let oy = -radius; oy <= radius; oy += 1) {
        for (let ox = -radius; ox <= radius; ox += 1) mark(x0 + ox, y0 + oy);
      }
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  };
  if (kind === "road") {
    line(34, 140, 102, 48, 3);
    line(186, 140, 118, 48, 3);
    line(14, 22, 206, 22, 2);
    line(24, 140, 92, 50, 2);
    line(196, 140, 128, 50, 2);
    for (let i = 0; i < 26; i += 1) mark(8 + ((i * 37) % 204), 8 + ((i * 53) % 134));
  } else {
    line(34, 30, 186, 120, 2);
    line(186, 30, 34, 120, 2);
    line(60, 60, 160, 60, 2);
    line(60, 60, 60, 100, 2);
    line(160, 60, 160, 100, 2);
    line(60, 100, 160, 100, 2);
    for (let i = 0; i < 12; i += 1) mark(10 + ((i * 29) % 200), 10 + ((i * 47) % 130));
  }
  return { edges, width, height };
}

function houghVote(scene, thetaStepDeg, rhoRes) {
  const { edges, width, height } = scene;
  const rhoMax = Math.hypot(width, height);
  const numTheta = Math.floor(180 / thetaStepDeg);
  const numRho = Math.floor((2 * rhoMax) / rhoRes) + 1;
  const accumulator = new Float32Array(numRho * numTheta);
  const cosTable = new Float32Array(numTheta);
  const sinTable = new Float32Array(numTheta);
  for (let t = 0; t < numTheta; t += 1) {
    const angle = (t * thetaStepDeg * Math.PI) / 180;
    cosTable[t] = Math.cos(angle);
    sinTable[t] = Math.sin(angle);
  }
  for (let y = 0; y < height; y += 1) {
    const row = edges[y];
    for (let x = 0; x < width; x += 1) {
      if (!row[x]) continue;
      for (let t = 0; t < numTheta; t += 1) {
        const rho = x * cosTable[t] + y * sinTable[t];
        const r = Math.round((rho + rhoMax) / rhoRes);
        if (r >= 0 && r < numRho) accumulator[r * numTheta + t] += 1;
      }
    }
  }
  return { accumulator, numRho, numTheta, rhoMax, rhoRes, thetaStepDeg };
}

function houghPeaks(state, thresholdPercent) {
  const { accumulator, numRho, numTheta } = state;
  let maxVotes = 0;
  for (let i = 0; i < accumulator.length; i += 1) {
    if (accumulator[i] > maxVotes) maxVotes = accumulator[i];
  }
  if (maxVotes <= 0) return { peaks: [], maxVotes: 0 };
  const working = new Float32Array(accumulator);
  const threshold = maxVotes * (thresholdPercent / 100);
  const peaks = [];
  for (let attempt = 0; attempt < 12; attempt += 1) {
    let best = -1;
    let bestValue = -1;
    for (let i = 0; i < working.length; i += 1) {
      if (working[i] > bestValue) { bestValue = working[i]; best = i; }
    }
    if (best < 0 || bestValue < threshold) break;
    const r = Math.floor(best / numTheta);
    const t = best % numTheta;
    peaks.push({ r, t, votes: bestValue });
    for (let dr = -3; dr <= 3; dr += 1) {
      for (let dt = -2; dt <= 2; dt += 1) {
        const nr = r + dr;
        const nt = t + dt;
        if (nr >= 0 && nr < numRho && nt >= 0 && nt < numTheta) working[nr * numTheta + nt] = 0;
      }
    }
  }
  return { peaks, maxVotes };
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let rgb;
  if (h < 60) rgb = [c, x, 0];
  else if (h < 120) rgb = [x, c, 0];
  else if (h < 180) rgb = [0, c, x];
  else if (h < 240) rgb = [0, x, c];
  else if (h < 300) rgb = [x, 0, c];
  else rgb = [c, 0, x];
  return rgb.map((v) => Math.round((v + m) * 255));
}

function drawHoughAccumulator(canvas, state, selectedIndex) {
  const ctx = canvas.getContext("2d");
  const { accumulator, numRho, numTheta, rhoMax, rhoRes } = state;
  const { width, height } = canvas;
  let maxVotes = 0;
  for (let i = 0; i < accumulator.length; i += 1) {
    if (accumulator[i] > maxVotes) maxVotes = accumulator[i];
  }
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  for (let py = 0; py < height; py += 1) {
    const r = Math.min(numRho - 1, Math.floor((py / height) * numRho));
    for (let px = 0; px < width; px += 1) {
      const t = Math.min(numTheta - 1, Math.floor((px / width) * numTheta));
      const value = maxVotes > 0 ? accumulator[r * numTheta + t] / maxVotes : 0;
      const [red, green, blue] = hslToRgb(168, 0.72, 0.94 - value * 0.58);
      const offset = (py * width + px) * 4;
      data[offset] = red;
      data[offset + 1] = green;
      data[offset + 2] = blue;
      data[offset + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  if (selectedIndex != null && selectedIndex >= 0) {
    const peak = houghCurrentPeaks[selectedIndex];
    if (peak) {
      const x = ((peak.t + 0.5) / numTheta) * width;
      const y = ((peak.r + 0.5) / numRho) * height;
      ctx.strokeStyle = "#e28c3c";
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 3, y - 3, 6, 6);
    }
  }
}

function drawHoughLineOnCanvas(canvas, rho, thetaDeg) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const theta = (thetaDeg * Math.PI) / 180;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const points = [];
  if (Math.abs(sin) > 0.02) {
    for (const x of [0, width]) {
      const y = (rho - x * cos) / sin;
      if (y >= -0.5 && y <= height + 0.5) points.push([x, y]);
    }
  }
  if (Math.abs(cos) > 0.02) {
    for (const y of [0, height]) {
      const x = (rho - y * sin) / cos;
      if (x >= -0.5 && x <= width + 0.5) points.push([x, y]);
    }
  }
  if (points.length === 2) {
    ctx.strokeStyle = "#e04f2f";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    ctx.lineTo(points[1][0], points[1][1]);
    ctx.stroke();
  }
}

function drawHoughEdges(canvas, scene) {
  const ctx = canvas.getContext("2d");
  const { edges, width, height } = scene;
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (edges[y][x]) {
        data[offset] = 38;
        data[offset + 1] = 49;
        data[offset + 2] = 46;
      } else {
        data[offset] = 247;
        data[offset + 1] = 249;
        data[offset + 2] = 248;
      }
      data[offset + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

let houghState = null;
let houghCurrentPeaks = [];
let houghSelectedIndex = -1;

function renderHough() {
  const scene = houghScene(houghSceneKind);
  houghState = houghVote(scene, 2, 1);
  const thresholdPercent = Number(document.querySelector("#hough-threshold").value);
  const { peaks, maxVotes } = houghPeaks(houghState, thresholdPercent);
  houghCurrentPeaks = peaks;
  document.querySelector("#hough-threshold-value").textContent = `${thresholdPercent}%`;

  const edgeCanvas = document.querySelector("#hough-edge-canvas");
  drawHoughEdges(edgeCanvas, scene);
  houghCurrentPeaks.forEach((peak, index) => {
    const rho = peak.r * houghState.rhoRes - houghState.rhoMax;
    const thetaDeg = peak.t * houghState.thetaStepDeg;
    drawHoughLineOnCanvas(edgeCanvas, rho, thetaDeg);
    if (index === houghSelectedIndex) {
      const ctx = edgeCanvas.getContext("2d");
      ctx.strokeStyle = "#e28c3c";
      ctx.lineWidth = 3;
      const points = houghLineEndpoints(edgeCanvas.width, edgeCanvas.height, rho, thetaDeg);
      if (points) {
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        ctx.lineTo(points[1][0], points[1][1]);
        ctx.stroke();
      }
    }
  });
  drawHoughAccumulator(document.querySelector("#hough-accumulator-canvas"), houghState, houghSelectedIndex);

  if (houghSelectedIndex >= 0 && houghCurrentPeaks[houghSelectedIndex]) {
    const peak = houghCurrentPeaks[houghSelectedIndex];
    const rho = peak.r * houghState.rhoRes - houghState.rhoMax;
    document.querySelector("#hough-votes").textContent = String(peak.votes);
    document.querySelector("#hough-rho").textContent = rho.toFixed(1);
    document.querySelector("#hough-theta").textContent = (peak.t * houghState.thetaStepDeg).toFixed(0);
  } else {
    document.querySelector("#hough-votes").textContent = maxVotes > 0 ? String(maxVotes) : "0";
    document.querySelector("#hough-rho").textContent = "—";
    document.querySelector("#hough-theta").textContent = "—";
  }
}

function houghLineEndpoints(width, height, rho, thetaDeg) {
  const theta = (thetaDeg * Math.PI) / 180;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const points = [];
  if (Math.abs(sin) > 0.02) {
    for (const x of [0, width]) {
      const y = (rho - x * cos) / sin;
      if (y >= -0.5 && y <= height + 0.5) points.push([x, y]);
    }
  }
  if (Math.abs(cos) > 0.02) {
    for (const y of [0, height]) {
      const x = (rho - y * sin) / cos;
      if (x >= -0.5 && x <= width + 0.5) points.push([x, y]);
    }
  }
  return points.length === 2 ? points : null;
}

let houghSceneKind = "road";
document.querySelectorAll("[data-hough-scene]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-hough-scene]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    houghSceneKind = button.dataset.houghScene;
    houghSelectedIndex = -1;
    renderHough();
  });
});
document.querySelector("#hough-threshold").addEventListener("input", () => {
  houghSelectedIndex = -1;
  renderHough();
});
document.querySelector("#hough-accumulator-canvas").addEventListener("click", (event) => {
  if (!houghState) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const px = (event.clientX - rect.left) * (event.currentTarget.width / rect.width);
  const py = (event.clientY - rect.top) * (event.currentTarget.height / rect.height);
  const t = Math.min(houghState.numTheta - 1, Math.floor((px / event.currentTarget.width) * houghState.numTheta));
  const r = Math.min(houghState.numRho - 1, Math.floor((py / event.currentTarget.height) * houghState.numRho));
  const target = r * houghState.numTheta + t;
  let best = -1;
  let bestDistance = Infinity;
  houghCurrentPeaks.forEach((peak, index) => {
    const distance = Math.abs(peak.r - r) + Math.abs(peak.t - t) * 0.4;
    if (distance < bestDistance) { bestDistance = distance; best = index; }
  });
  if (best >= 0 && bestDistance < 8) {
    houghSelectedIndex = best;
    renderHough();
  }
});
renderHough();

// ---------- Lucas-Kanade optical flow lab ----------

function buildFlowFrames(width, height, dx, dy) {
  const a = new Float32Array(width * height);
  const b = new Float32Array(width * height);
  const paintBackground = (frame) => {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        frame[y * width + x] = 42 + 0.25 * x + 2 * Math.sin((2 * Math.PI * y) / 36) + 2 * Math.sin((2 * Math.PI * (x + y)) / 28);
      }
    }
  };
  const paintRect = (frame, ox, oy) => {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (x >= 56 + ox && x < 104 + ox && y >= 70 + oy && y < 118 + oy) {
          const rx = x - ox;
          const ry = y - oy;
          frame[y * width + x] = 205 + 10 * Math.sin((2 * Math.PI * (rx + ry)) / 36) + 6 * Math.sin((2 * Math.PI * (rx + 2 * ry)) / 30);
        }
      }
    }
  };
  const paintCircle = (frame, ox, oy) => {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const radius2 = (x - (170 + ox)) ** 2 + (y - (60 + oy)) ** 2;
        if (radius2 <= 18 * 18) {
          const rx = x - ox;
          const ry = y - oy;
          frame[y * width + x] = 32 + 8 * Math.sin((2 * Math.PI * (rx + ry)) / 30) + 4 * Math.sin((2 * Math.PI * (rx - ry)) / 24);
        }
      }
    }
  };
  paintBackground(a);
  paintBackground(b);
  paintRect(a, 0, 0);
  paintCircle(a, 0, 0);
  paintRect(b, dx, dy);
  paintCircle(b, -dx, dy);
  return { a, b, width, height };
}

function computeFlowLite(frames, windowSize, step) {
  const { a, b, width, height } = frames;
  const half = Math.floor(windowSize / 2);
  const vectors = [];
  const average = (x, y) => 0.5 * (a[y * width + x] + b[y * width + x]);
  for (let y = half + 3; y < height - half - 3; y += step) {
    for (let x = half + 3; x < width - half - 3; x += step) {
      let a11 = 0;
      let a12 = 0;
      let a22 = 0;
      let b1 = 0;
      let b2 = 0;
      for (let wy = -half; wy <= half; wy += 1) {
        for (let wx = -half; wx <= half; wx += 1) {
          const cx = x + wx;
          const cy = y + wy;
          const gx = 0.5 * (average(cx + 1, cy) - average(cx - 1, cy));
          const gy = 0.5 * (average(cx, cy + 1) - average(cx, cy - 1));
          const gt = b[cy * width + cx] - a[cy * width + cx];
          a11 += gx * gx;
          a12 += gx * gy;
          a22 += gy * gy;
          b1 += gx * gt;
          b2 += gy * gt;
        }
      }
      const det = a11 * a22 - a12 * a12;
      const trace = a11 + a22;
      const eigen = 0.5 * (trace - Math.sqrt((a11 - a22) ** 2 + 4 * a12 * a12));
      if (det > 0.001 && eigen > 8) {
        const u = (-a22 * b1 + a12 * b2) / det;
        const v = (a12 * b1 - a11 * b2) / det;
        vectors.push({ x, y, u, v, confidence: eigen });
      }
    }
  }
  return vectors;
}

function drawFlowCanvas(canvas, frames, vectors, dx, dy) {
  const ctx = canvas.getContext("2d");
  const { b, width, height } = frames;
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = Math.max(0, Math.min(255, b[y * width + x]));
      const offset = (y * width + x) * 4;
      data[offset] = value;
      data[offset + 1] = value;
      data[offset + 2] = value;
      data[offset + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  let rectCount = 0;
  let rectSpeedSum = 0;
  vectors.forEach((vector) => {
    const speed = Math.hypot(vector.u, vector.v);
    if (speed < 0.25) return;
    const inRect = vector.x >= 56 + dx && vector.x < 104 + dx && vector.y >= 70 + dy && vector.y < 118 + dy;
    if (inRect) { rectCount += 1; rectSpeedSum += speed; }
    const color = vector.confidence > 80 ? "#26ae60" : "#e28c3c";
    drawArrow(ctx, vector.x, vector.y, vector.x + vector.u * 3, vector.y + vector.v * 3, color, 1.6);
  });
  document.querySelector("#flow-valid-count").textContent = String(vectors.length);
  document.querySelector("#flow-rect-count").textContent = String(rectCount);
  document.querySelector("#flow-rect-speed").textContent = rectCount > 0 ? `${(rectSpeedSum / rectCount).toFixed(2)} px` : "—";
}

function drawArrow(ctx, x1, y1, x2, y2, color, width) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.hypot(dx, dy);
  if (length < 1e-6) return;
  const ux = dx / length;
  const uy = dy / length;
  const head = 4.5;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * ux - head * 0.5 * uy, y2 - head * uy + head * 0.5 * ux);
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * ux + head * 0.5 * uy, y2 - head * uy - head * 0.5 * ux);
  ctx.stroke();
}

function renderFlowLab() {
  const dx = Number(document.querySelector("#flow-dx").value);
  const dy = Number(document.querySelector("#flow-dy").value);
  const windowSize = Number(document.querySelector("#flow-window").value);
  document.querySelector("#flow-dx-value").textContent = `${dx} px`;
  document.querySelector("#flow-dy-value").textContent = `${dy} px`;
  document.querySelector("#flow-window-value").textContent = `${windowSize} px`;
  const frames = buildFlowFrames(260, 190, dx, dy);
  const vectors = computeFlowLite(frames, windowSize, 6);
  drawFlowCanvas(document.querySelector("#flow-canvas"), frames, vectors, dx, dy);
}

["flow-dx", "flow-dy", "flow-window"].forEach((id) => {
  document.querySelector(`#${id}`).addEventListener("input", renderFlowLab);
});
renderFlowLab();
// ---------- Calibration distortion lab ----------
function calibDistort(x, y, k1, k2, p1, p2) {
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 * r2;
  return [
    x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
    y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
  ];
}

function drawCalibGrid(canvas, k1, k2, p1, p2, corrected) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const scale = Math.min(width, height) * 0.32;
  const cx = width / 2;
  const cy = height / 2;
  const cell = 0.23;
  const cols = 7;
  const rows = 5;
  const map = corrected
    ? (x, y) => [x, y]
    : (x, y) => calibDistort(x, y, k1, k2, p1, p2);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const x0 = (col - cols / 2) * cell;
      const y0 = (row - rows / 2) * cell;
      const corners = [
        map(x0, y0),
        map(x0 + cell, y0),
        map(x0 + cell, y0 + cell),
        map(x0, y0 + cell)
      ];
      ctx.beginPath();
      ctx.moveTo(cx + corners[0][0] * scale, cy + corners[0][1] * scale);
      for (let i = 1; i < corners.length; i += 1) {
        ctx.lineTo(cx + corners[i][0] * scale, cy + corners[i][1] * scale);
      }
      ctx.closePath();
      ctx.fillStyle = (col + row) % 2 === 0 ? "#f4f6f3" : "#22302d";
      ctx.fill();
      ctx.strokeStyle = "rgba(120, 134, 128, 0.45)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  ctx.strokeStyle = "#2f6f63";
  ctx.lineWidth = 1.5;
  for (let i = 0; i <= cols; i += 1) {
    const x = (i - cols / 2) * cell;
    ctx.beginPath();
    for (let step = 0; step <= 48; step += 1) {
      const y = (-rows / 2 + (step / 48) * rows) * cell;
      const [dx, dy] = map(x, y);
      if (step === 0) ctx.moveTo(cx + dx * scale, cy + dy * scale);
      else ctx.lineTo(cx + dx * scale, cy + dy * scale);
    }
    ctx.stroke();
  }
  for (let i = 0; i <= rows; i += 1) {
    const y = (i - rows / 2) * cell;
    ctx.beginPath();
    for (let step = 0; step <= 48; step += 1) {
      const x = (-cols / 2 + (step / 48) * cols) * cell;
      const [dx, dy] = map(x, y);
      if (step === 0) ctx.moveTo(cx + dx * scale, cy + dy * scale);
      else ctx.lineTo(cx + dx * scale, cy + dy * scale);
    }
    ctx.stroke();
  }
}

function renderCalibrationLab() {
  const k1 = Number(document.querySelector("#calib-k1").value);
  const k2 = Number(document.querySelector("#calib-k2").value);
  const p1 = Number(document.querySelector("#calib-p1").value);
  const p2 = Number(document.querySelector("#calib-p2").value);
  document.querySelector("#calib-k1-value").textContent = k1.toFixed(2);
  document.querySelector("#calib-k2-value").textContent = k2.toFixed(2);
  document.querySelector("#calib-p1-value").textContent = p1.toFixed(3);
  document.querySelector("#calib-p2-value").textContent = p2.toFixed(3);
  drawCalibGrid(document.querySelector("#calib-distorted-canvas"), k1, k2, p1, p2, false);
  drawCalibGrid(document.querySelector("#calib-corrected-canvas"), k1, k2, p1, p2, true);
}

["calib-k1", "calib-k2", "calib-p1", "calib-p2"].forEach((id) => {
  document.querySelector(`#${id}`).addEventListener("input", renderCalibrationLab);
});
renderCalibrationLab();

// ---------- PnP pose lab ----------
const pnpLandmarks = [
  [-3.0, -0.7, 8.5], [3.0, -0.7, 8.5],
  [-3.4, 0.5, 13.0], [3.4, 0.5, 13.0],
  [-3.8, 1.5, 19.0], [3.8, 1.5, 19.0],
  [-1.2, 0.1, 16.0], [1.2, 0.1, 16.0],
];

function projectPnPPoint(point, centerX, centerZ, yawDegrees) {
  const angle = yawDegrees * Math.PI / 180;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const dx = point[0] - centerX;
  const dz = point[2] - centerZ;
  const cameraX = cosine * dx + sine * dz;
  const cameraZ = -sine * dx + cosine * dz;
  if (cameraZ <= 0.2) return null;
  return {
    x: 210 + 360 * cameraX / cameraZ,
    y: 145 - 360 * point[1] / cameraZ,
    depth: cameraZ,
  };
}

function renderPnPLab() {
  const canvas = document.querySelector("#pnp-canvas");
  const ctx = canvas.getContext("2d");
  const centerX = Number(document.querySelector("#pnp-x").value);
  const centerZ = Number(document.querySelector("#pnp-z").value);
  const yaw = Number(document.querySelector("#pnp-yaw").value);
  document.querySelector("#pnp-x-value").textContent = `${centerX.toFixed(1)} m`;
  document.querySelector("#pnp-z-value").textContent = `${centerZ.toFixed(1)} m`;
  document.querySelector("#pnp-yaw-value").textContent = `${yaw}°`;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e8eeeb";
  ctx.fillRect(0, 0, 420, 310);
  ctx.fillStyle = "#d0d8d4";
  ctx.beginPath();
  ctx.moveTo(100, 310);
  ctx.lineTo(320, 310);
  ctx.lineTo(258, 62);
  ctx.lineTo(162, 62);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(155, 310);
  ctx.lineTo(187, 62);
  ctx.moveTo(265, 310);
  ctx.lineTo(233, 62);
  ctx.stroke();

  const projected = pnpLandmarks
    .map((point, index) => ({ point, index, pixel: projectPnPPoint(point, centerX, centerZ, yaw) }))
    .filter((item) => item.pixel && item.pixel.x >= 0 && item.pixel.x <= 420 && item.pixel.y >= 0 && item.pixel.y <= 310);
  projected.forEach(({ index, pixel }) => {
    const radius = Math.max(3, 9 - pixel.depth * 0.25);
    ctx.fillStyle = index % 2 === 0 ? "#319165" : "#bc6937";
    ctx.beginPath();
    ctx.arc(pixel.x, pixel.y, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  ctx.fillStyle = "#18211f";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("camera image: fixed 3-D landmarks move in pixels", 12, 328);

  const originX = 510;
  const bottomY = 292;
  ctx.strokeStyle = "#63706c";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(438, bottomY);
  ctx.lineTo(590, bottomY);
  ctx.moveTo(originX, 30);
  ctx.lineTo(originX, bottomY);
  ctx.stroke();
  pnpLandmarks.forEach((point, index) => {
    ctx.fillStyle = index % 2 === 0 ? "#319165" : "#bc6937";
    ctx.beginPath();
    ctx.arc(originX + point[0] * 18, bottomY - point[2] * 11, 3.5, 0, 2 * Math.PI);
    ctx.fill();
  });
  const cameraX = originX + centerX * 18;
  const cameraY = bottomY - centerZ * 11;
  ctx.fillStyle = "#bc6937";
  ctx.beginPath();
  ctx.moveTo(cameraX, cameraY - 9);
  ctx.lineTo(cameraX - 8, cameraY + 8);
  ctx.lineTo(cameraX + 8, cameraY + 8);
  ctx.closePath();
  ctx.fill();
  const heading = (yaw + 90) * Math.PI / 180;
  drawArrow(
    ctx,
    cameraX,
    cameraY,
    cameraX + Math.cos(heading) * 36,
    cameraY - Math.sin(heading) * 36,
    "#bc6937",
    2,
  );
  ctx.fillStyle = "#63706c";
  ctx.fillText("X–Z top view", 447, 328);

  const near = pnpLandmarks.slice(0, 2)
    .map((point) => projectPnPPoint(point, centerX, centerZ, yaw))
    .filter(Boolean);
  const far = pnpLandmarks.slice(4, 6)
    .map((point) => projectPnPPoint(point, centerX, centerZ, yaw))
    .filter(Boolean);
  const spread = (pair) => pair.length === 2 ? Math.abs(pair[1].x - pair[0].x) : null;
  const nearSpread = spread(near);
  const farSpread = spread(far);
  document.querySelector("#pnp-visible-count").textContent = String(projected.length);
  document.querySelector("#pnp-near-spread").textContent = nearSpread === null ? "—" : `${nearSpread.toFixed(0)} px`;
  document.querySelector("#pnp-far-spread").textContent = farSpread === null ? "—" : `${farSpread.toFixed(0)} px`;
}

["pnp-x", "pnp-z", "pnp-yaw"].forEach((id) => {
  document.querySelector(`#${id}`).addEventListener("input", renderPnPLab);
});
renderPnPLab();

// ---------- Visual-odometry drift lab ----------
function buildDriftTrajectory(steps, yawBiasDegrees, noiseMagnitude) {
  const truth = [{ x: 0, z: 0 }];
  const estimate = [{ x: 0, z: 0 }];
  let heading = 0;
  for (let index = 1; index <= steps; index += 1) {
    truth.push({ x: 0, z: index * 0.8 });
    heading += yawBiasDegrees * Math.PI / 180;
    const deterministicXNoise = Math.sin(index * 12.9898) * noiseMagnitude;
    const deterministicZNoise = Math.sin(index * 7.233 + 1.7) * noiseMagnitude;
    const previous = estimate[estimate.length - 1];
    estimate.push({
      x: previous.x + Math.sin(heading) * 0.8 + deterministicXNoise,
      z: previous.z + Math.cos(heading) * 0.8 + deterministicZNoise,
    });
  }
  return { truth, estimate, heading };
}

function renderVODriftLab() {
  const yawBias = Number(document.querySelector("#vo-yaw").value);
  const steps = Number(document.querySelector("#vo-steps").value);
  const noise = Number(document.querySelector("#vo-noise").value);
  document.querySelector("#vo-yaw-value").textContent = `${yawBias.toFixed(2)}°`;
  document.querySelector("#vo-steps-value").textContent = String(steps);
  document.querySelector("#vo-noise-value").textContent = `${noise.toFixed(2)} m`;

  const trajectory = buildDriftTrajectory(steps, yawBias, noise);
  const canvas = document.querySelector("#vo-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e8eeeb";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const all = [...trajectory.truth, ...trajectory.estimate];
  const xs = all.map((point) => point.x);
  const zs = all.map((point) => point.z);
  const minX = Math.min(...xs, -1);
  const maxX = Math.max(...xs, 1);
  const minZ = Math.min(...zs, 0);
  const maxZ = Math.max(...zs, 1);
  const padding = 34;
  const spanX = Math.max(maxX - minX, 2);
  const spanZ = Math.max(maxZ - minZ, 2);
  const scale = Math.min(
    (canvas.width - 2 * padding) / spanX,
    (canvas.height - 2 * padding) / spanZ,
  );
  const offsetX = canvas.width / 2 - ((minX + maxX) / 2) * scale;
  const map = (point) => ({
    x: offsetX + point.x * scale,
    y: canvas.height - padding - (point.z - minZ) * scale,
  });

  ctx.strokeStyle = "rgba(99, 112, 108, 0.25)";
  ctx.lineWidth = 1;
  for (let index = 0; index < trajectory.truth.length; index += Math.max(1, Math.floor(steps / 12))) {
    const truthPoint = map(trajectory.truth[index]);
    const estimatePoint = map(trajectory.estimate[index]);
    ctx.beginPath();
    ctx.moveTo(truthPoint.x, truthPoint.y);
    ctx.lineTo(estimatePoint.x, estimatePoint.y);
    ctx.stroke();
  }

  function drawTrack(points, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach((point, index) => {
      const screen = map(point);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    });
    ctx.stroke();
  }

  drawTrack(trajectory.truth, "#319165", 4);
  drawTrack(trajectory.estimate, "#bc6937", 3);
  const truthEnd = map(trajectory.truth[trajectory.truth.length - 1]);
  const estimateEnd = map(trajectory.estimate[trajectory.estimate.length - 1]);
  ctx.fillStyle = "#319165";
  ctx.beginPath();
  ctx.arc(truthEnd.x, truthEnd.y, 5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#bc6937";
  ctx.beginPath();
  ctx.arc(estimateEnd.x, estimateEnd.y, 5, 0, 2 * Math.PI);
  ctx.fill();

  const squaredErrors = trajectory.truth.map((point, index) => {
    const dx = trajectory.estimate[index].x - point.x;
    const dz = trajectory.estimate[index].z - point.z;
    return dx * dx + dz * dz;
  });
  const ate = Math.sqrt(squaredErrors.reduce((sum, value) => sum + value, 0) / squaredErrors.length);
  const final = trajectory.estimate[trajectory.estimate.length - 1];
  document.querySelector("#vo-final-drift").textContent = `${Math.abs(final.x).toFixed(2)} m`;
  document.querySelector("#vo-ate").textContent = `${ate.toFixed(2)} m`;
  document.querySelector("#vo-heading-error").textContent = `${Math.abs(yawBias * steps).toFixed(1)}°`;
}

["vo-yaw", "vo-steps", "vo-noise"].forEach((id) => {
  document.querySelector(`#${id}`).addEventListener("input", renderVODriftLab);
});
renderVODriftLab();
