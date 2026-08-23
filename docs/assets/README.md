# Asset provenance

Algorithm result images in this directory are generated deterministically by:

```bash
python -m vision2autonomy.examples.reference_images
```

`autonomous_driving_cv_map.png` is a conceptual teaching illustration generated for this repository with OpenAI's built-in image generation tool on 2026-08-23. It is not a camera dataset sample and must not be used for quantitative evaluation.

Prompt summary: a restrained forward-facing urban driving scene with sparse teal lane/object contours, amber stable feature points, cross-frame motion vectors, and translucent depth bands; no text, logos, watermark, or futuristic HUD clutter.

The illustration communicates where algorithms can contribute. It does not claim to represent the output of a production autonomous-driving stack.
