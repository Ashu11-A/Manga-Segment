# Manga-Segment — unified CV framework

A single [uv](https://docs.astral.sh/uv/)-managed package that bundles **training**,
**inference** and **serving** for manga segmentation behind one scalable,
plug-in framework. It replaces the previous split `src/` (Flask proxy) and
`training/` (YOLO + U-Net) trees.

Two algorithms ship today — **YOLO** (Ultralytics: train / tune / fine-tune /
convert / benchmark / inference) and **U-Net** (pre-trained inference + the
legacy Keras training pipeline) — and new computer-vision algorithms can be added
without touching the CLI or serving layer.

## Setup

```bash
cd src
uv sync                       # inference + serving + YOLO/U-Net inference & training
uv sync --extra unet-training # also install tensorflowjs + keras-tuner (U-Net train/convert)
```

Run every command through `uv run` so it executes inside the managed environment.

## Layout

```
src/
  main.py            # unified subcommand CLI
  core/              # the framework (no heavy imports at import time)
    paths.py         #   repo-root-relative paths (models/, images/, output/, dataset/)
    device.py        #   resolve_device / cuda_device_count / supports_half
    imaging.py       #   IMAGE_EXTS, image IO, composite_rgba, combine_masks
    segmenter.py     #   Instance, SegmentationResult, BaseSegmenter (directory workflow)
    algorithm.py     #   BaseAlgorithm contract + NotSupported
    registry.py      #   register / available / get_algorithm
  algorithms/        # plug-ins (auto-discovered by load_all)
    yolo/            #   algorithm.py, segmenter.py, weights.py, benchmark.py, best_hyperparameters.yaml
    unet/            #   algorithm.py, segmenter.py, architecture.py, data.py, preprocess.py
  serving/           # Flask proxy (app factory) reusing the framework
```

## CLI

### Interactive mode (no arguments)

Run `main.py` with **no arguments** to launch an arrow-key wizard that walks you
through every command and option, then prints the equivalent command line and
runs it (for `train`, training starts immediately):

```bash
uv run python main.py        # ↑/↓ pick command → space-toggle options → fill in → run
```

The wizard introspects the parser, so it always offers exactly the same commands
and flags as the explicit CLI below. Anything you leave untouched keeps its
default. Every explicit invocation still works unchanged:

```bash
uv run python main.py train     --algo yolo  --size 1280
uv run python main.py train     --algo yolo  --model 1            # resume/fine-tune base model
uv run python main.py tune      --algo yolo
uv run python main.py convert   --algo yolo  --model 1            # export ONNX/TFJS
uv run python main.py benchmark --algo yolo  --workers 2          # train families + report
uv run python main.py report    --algo yolo

uv run python main.py test --algo yolo --model-dir ../models/yolo
uv run python main.py test --algo unet --model-dir ../models/unet --threshold 0.5
uv run python main.py serve --algo yolo --model-dir ../models/yolo --port 5000
```

### Unified trained-model selection (`test` / `convert` / `serve`)

| Flag | Meaning |
|--|--|
| `--model-dir <path>` | Explicit trained model (highest priority). YOLO: a `.pt`, a run dir like `../models/yolo`, or a `weights/` dir. U-Net: a `SavedModel` directory. |
| `--model <id>` | YOLO: `train<id>` under `../runs/segment`. U-Net: `../models/unet-<id>`. |
| neither | YOLO: most recent `train*` run. U-Net: `../models/unet`. |

### Inference outputs (`test`)

For every input image in `--images` (default `../images`), results go to `--output`
(default `../output`):

| File | Source |
|--|--|
| `<name>_<class>_<i>_mask.png` | YOLO per-instance grayscale mask |
| `<name>_unet_mask.png` | U-Net raw predicted RGBA mask |
| `<name>_segmented.png` | Original image with the background removed (transparent PNG) |

Compositing (`core.imaging.composite_rgba`) and the directory workflow
(`core.segmenter.BaseSegmenter`) are shared by both backends.

## Serving

```bash
uv run python main.py serve --algo yolo --model-dir ../models/yolo --port 5000
```

`GET /image?url=<image-url>&mode=<segmented|annotated>`:

- `segmented` (default) — transparent-background PNG via `segmenter.segment_array`.
- `annotated` — the detections drawn on the image (YOLO `result.plot()`).

Designed for the [Bandwidth Hero](https://bandwidth-hero.com/) proxy, like the
original `src/app.py`.

## Adding a new algorithm

1. Create `algorithms/<name>/`.
2. Implement a `BaseSegmenter` subclass (only `predict(image_bgr) -> SegmentationResult`).
3. Implement a `BaseAlgorithm` subclass decorated with `@register` (`resolve_model_ref`
   + `build_segmenter`; override `train`/`tune`/`convert`/`benchmark`/`report` as
   supported — unimplemented ones raise `NotSupported` automatically).
4. Export the algorithm from the sub-package `__init__.py`.

`load_all()` auto-discovers the sub-package; the CLI (`--algo <name>`) and serving
pick it up with no further wiring.
