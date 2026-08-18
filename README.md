# Vision2Autonomy

Computer vision from first principles to autonomous-driving perception.

Vision2Autonomy is a progressive, test-driven learning library. It begins with
the numerical foundations of image processing, develops classical and learned
vision algorithms, and culminates in a camera-based driving-perception system.

## Why this repository exists

Each topic is developed at three levels:

1. **Understand** — concise notes explain the mathematics and design choices.
2. **Implement** — core algorithms are written from scratch with NumPy before
   higher-level libraries are introduced.
3. **Verify** — unit tests, reference comparisons, and experiments establish
   correctness rather than relying only on attractive output images.

## Current milestone: Canny from scratch

The first working pipeline includes:

- vectorized 2-D convolution with explicit padding modes;
- Gaussian kernel generation and smoothing;
- Sobel gradients, magnitude, and orientation;
- non-maximum suppression;
- double thresholding and 8-connected hysteresis;
- an inspectable result containing every intermediate stage;
- a command-line interface and automated tests.

The implementation does **not** call `cv2.Canny`.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest
```

Run the detector on an image:

```bash
v2a-canny input.jpg edges.png --low 40 --high 100
```

Or use the Python API:

```python
from PIL import Image
import numpy as np

from vision2autonomy import canny

image = np.asarray(Image.open("input.jpg").convert("L"))
edges = canny(image, low_threshold=40, high_threshold=100)
Image.fromarray(edges).save("edges.png")
```

## Project layout

```text
src/vision2autonomy/    reusable library code
chapters/               progressive learning material
tests/                  correctness and integration tests
docs/roadmap.md         path toward driving perception
assignment/             original 2016 coursework, retained for history
```

See the [roadmap](docs/roadmap.md) for upcoming chapters.

## Case documentation contract

Every case must include an executable reproduction command, its exact input,
meaningful intermediate images, a final reference image, parameter guidance,
result interpretation, known limitations, and an automated test. Generated
figures are committed under `docs/assets/` and must be reproducible from code.

See the complete [case documentation standard](docs/CASE_STANDARD.md). The
current reference figures can be regenerated with:

```bash
python -m vision2autonomy.examples.reference_images
```

## Project history

This repository began as a 2016 computer-vision assignment. Its original code
is intentionally retained while the project is rebuilt as a modern, tested
learning library. The commit history documents that evolution.
