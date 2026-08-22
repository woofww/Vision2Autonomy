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

const themeToggle = document.querySelector("#theme-toggle");
const storedTheme = localStorage.getItem("v2a-theme");
if (storedTheme) document.documentElement.dataset.theme = storedTheme;
themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("v2a-theme", next);
});
