# CLI Commands

This document describes the command-line interfaces available in Manga-Segment.
Examples assume they are run from the repository root:

```bash
uv run python src/main.py <command> [options]
```

If you are already inside `src/`, use `uv run python main.py <command> [options]` instead.

## Main CLI

Running the main CLI with no arguments opens an interactive wizard:

```bash
uv run python src/main.py
```

The wizard lets you choose a command and its options, prints the equivalent command line, and then executes it.

### Common Algorithm Parameters

These parameters are shared by `train`, `test`, `tune`, `convert`, `benchmark`, `report`, and `serve`.

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm backend to use. Default: `yolo`. Registered backends include `yolo` and `unet`; unsupported operations return a friendly error. |
| `--size` | Input image size. If omitted, training commands auto-detect the size from the dataset when possible. For inference, YOLO uses the trained model size when available, otherwise `1280`. |

### Common Model Selection Parameters

These parameters are shared by `test`, `convert`, `report`, and `serve`.

| Parameter | Description |
| --- | --- |
| `--model` | Numeric trained model id. For YOLO, this selects `runs/segment/train<N>`; `0` selects the unsuffixed `runs/segment/train`. For U-Net, this selects `models/unet-<N>`. |
| `--model-dir` | Explicit trained model path. For YOLO, this can be a `.pt` file, a run directory, or a `weights/` directory. For U-Net, this is a SavedModel directory. This option has priority over `--model`. |

For YOLO, `--model-dir models/yolo` searches for checkpoints in this order:

```text
models/yolo/weights/best.pt
models/yolo/best.pt
```

To use a checkpoint with another name, point directly to it:

```bash
uv run python src/main.py test --model-dir models/yolo/weights/last.pt
```

## `train`

Trains a model from scratch, fine-tunes an existing model with `--model`, or resumes an interrupted run with `--resume`.

```bash
uv run python src/main.py train --algo yolo --size 1400
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm to train. Default: `yolo`. |
| `--size` | Training image size. If omitted, it is auto-detected from the dataset images when supported. |
| `--model` | Existing model id to resume or use as the base for fine-tuning. For YOLO, this resolves to `runs/segment/train<N>`. |
| `--epochs` | Number of training epochs. YOLO default when omitted: `1000`; U-Net default when omitted: `250`. |
| `--batch` | Training batch size. YOLO default when omitted: `8`. Also used when resuming YOLO after an out-of-memory failure. |
| `--patience` | Early-stopping patience. YOLO default when omitted: `100`. |
| `--resume` | Resume an interrupted YOLO run from `last.pt`. Use `--model <id>` to choose the run; omit `--model` to resume the latest run. |
| `--base-model` | Base YOLO segmentation weights used when training from scratch. Default: `yolo26s-seg.pt`. Choices: `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`, `yolov9t-seg.pt`, `yolov9s-seg.pt`, `yolov9m-seg.pt`, `yolov9c-seg.pt`, `yolov9e-seg.pt`, `yolov10n-seg.pt`, `yolov10s-seg.pt`, `yolov10m-seg.pt`, `yolov10b-seg.pt`, `yolov10l-seg.pt`, `yolov10x-seg.pt`, `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt`, `yolo12n-seg.pt`, `yolo12s-seg.pt`, `yolo12m-seg.pt`, `yolo12l-seg.pt`, `yolo12x-seg.pt`, `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt`. |

## `test`

Runs segmentation inference over a folder of images.

```bash
uv run python src/main.py test --algo yolo --model-dir models/yolo --conf 0.5
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm to use for inference. Default: `yolo`. Use `unet` for U-Net inference. |
| `--size` | Inference image size. For YOLO, this overrides the trained image size. If omitted, YOLO reads the saved training size when possible or falls back to `1280`. |
| `--model` | Numeric trained model id. For YOLO, selects `runs/segment/train<N>`; for U-Net, selects `models/unet-<N>`. |
| `--model-dir` | Explicit model path. YOLO accepts a `.pt` file, a run directory, or a `weights/` directory. U-Net accepts a SavedModel directory. |
| `--images` | Input image directory. Default: `images/` at the repository root. |
| `--output` | Output directory. Default: `output/` at the repository root. |
| `--threshold` | U-Net mask threshold. Default: `0.5`. |
| `--conf` | YOLO confidence threshold. Default: `0.5`. |
| `--keep-classes` | Allow-list of classes to keep in the final output. Provide one or more class names, for example `--keep-classes speech-balloon thought-balloon`. Default: keep all classes. |
| `--ignore-classes` | Deny-list of classes to remove from the final output. Provide one or more class names, for example `--ignore-classes text`. Default: ignore no classes. |
| `--only-segmented` | Export only `<name>_segmented.png` for each image. This is enabled by default. |
| `--no-only-segmented` | Disable `--only-segmented` and also export per-instance mask files. |
| `--draw` | Export only a colored website-demo-style overlay as `<name>_overlay.png`. This respects `--keep-classes` and `--ignore-classes`, and it does not write masks or segmented composites. |

Output files for `test`:

| File | Description |
| --- | --- |
| `<name>_segmented.png` | Original image composited with the background removed. This is the default YOLO output. |
| `<name>_<class>_<i>_mask.png` | YOLO per-instance mask file. Written only with `--no-only-segmented`. |
| `<name>_unet_mask.png` | U-Net predicted mask. |
| `<name>_overlay.png` | Colored YOLO overlay. Written only with `--draw`, and it is the only file written in draw mode. |

Examples based on `Comands.md`:

```bash
# Default inference
uv run python src/main.py test --model-dir models/yolo --conf 0.5

# Custom folders
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --images ./pages --output ./results

# Overlay only
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --draw

# Remove text from the composite
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --ignore-classes text

# Keep only balloons
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --keep-classes speech-balloon thought-balloon

# Segmented PNG only; this is the default behavior
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --only-segmented

# Segmented PNG plus per-instance masks
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --no-only-segmented

# Force inference image size
uv run python src/main.py test --model-dir models/yolo --conf 0.5 --size 1024
```

## `tune`

Runs hyperparameter tuning for the selected algorithm.

```bash
uv run python src/main.py tune --algo yolo
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm to tune. Default: `yolo`. |
| `--size` | Image size used during tuning. If omitted, it is auto-detected from the dataset when supported. |
| `--base-model` | Base YOLO segmentation weights to tune. Default: `yolo26s-seg.pt`. Accepts the same choices as `train --base-model`. |

## `convert`

Exports a trained model to a deployment format.

```bash
uv run python src/main.py convert --algo yolo --model 10
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm to convert. Default: `yolo`. YOLO exports to ONNX; U-Net exports a SavedModel to TensorFlow.js. |
| `--size` | Export image size override. For YOLO, this overrides the size read from the checkpoint metadata. |
| `--model` | Numeric trained model id to convert. |
| `--model-dir` | Explicit model path to convert. Has priority over `--model`. |

## `benchmark`

Trains several YOLO model families and writes a benchmark report.

```bash
uv run python src/main.py benchmark --algo yolo --workers 2
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm to benchmark. Default: `yolo`. |
| `--size` | Training image size. If omitted, it is auto-detected from the dataset. |
| `--workers` | Number of parallel training jobs. Default: number of available GPUs. |
| `--epochs` | Epochs per benchmark training run. Default: `1000`. |
| `--patience` | Early-stopping patience for each run. Default: `100`. |
| `--batch` | Batch size for each run. Default: `5`. |

## `report`

Generates dataset, model metadata, and validation/test metric reports.

```bash
uv run python src/main.py report --algo yolo
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm to report on. Default: `yolo`. |
| `--size` | Image size used for report evaluation when needed. |
| `--model` | Numeric trained model id to evaluate. |
| `--model-dir` | Explicit model path to evaluate. Has priority over `--model`. |
| `--data` | Explicit dataset `data.yaml` path. If omitted, the project dataset discovery is used. |
| `--output` | Write a single Markdown report to this file. |
| `--output-dir` | Directory for the default separate report files. Default: `reports/` at the repository root. |
| `--dataset-only` | Generate only dataset statistics, without model metadata or validation/test evaluation. |

Default YOLO report files are `reports/yolo_metadata.md` and `reports/yolo_metrics.md`.

## `serve`

Starts the HTTP segmentation proxy.

```bash
uv run python src/main.py serve --algo yolo --model-dir models/yolo --port 5000
```

| Parameter | Description |
| --- | --- |
| `--algo` | Algorithm backend used by the server. Default: `yolo`. |
| `--size` | Inference image size override. |
| `--model` | Numeric trained model id to serve. |
| `--model-dir` | Explicit trained model path to serve. Has priority over `--model`. |
| `--host` | Host interface to bind. Default: `127.0.0.1`. |
| `--port` | TCP port to bind. Default: `5000`. |
| `--mode` | Default response mode for `/image`. Choices: `segmented` or `annotated`. Default: `segmented`. |

The server exposes:

```text
GET /image?url=<image-url>&mode=<segmented|annotated>
```

`mode=segmented` returns a transparent-background PNG. `mode=annotated` returns an annotated detection image.

## `download`

Runs an interactive Roboflow dataset download wizard and can start YOLO training after the download.

```bash
uv run python src/main.py download --version 5 --model yolo26s-seg.pt
```

Required environment variable: `ROBOFLOW_API_KEY`. Optional environment variables: `ROBOFLOW_WORKSPACE` and `ROBOFLOW_PROJECT`.

| Parameter | Description |
| --- | --- |
| `--version` | Roboflow dataset version number. When provided, skips the version prompt. |
| `--model` | Base YOLO weights file, for example `yolo26s-seg.pt`. When provided, skips the model family and size prompts. The model name is also used to infer the Roboflow export format. |
| `--skip-train` | Download the dataset only and do not start training. |
| `--size` | Image size used if training starts after download. If omitted, training auto-detects the dataset image size when supported. |

## `distill`

Distills a teacher text detector into a YOLO segmentation dataset, mainly for generating text masks.

```bash
uv run python src/main.py distill --teacher ensemble --source manga-segment.v20i.yolo26 --dest manga-segment-text.v20i.yolo
```

If PaddleOCR is unavailable in the main project environment, use the isolated script with the same parameters:

```bash
uv run python src/dataset/run.py --teacher ensemble --source manga-segment.v20i.yolo26 --dest manga-segment-text.v20i.yolo
```

| Parameter | Description |
| --- | --- |
| `--teacher` | Teacher detector to use. Default: `ensemble`. Accepted values include `ensemble`, `paddleocr`, `ctd`, or a comma-separated list such as `paddleocr,ctd`. |
| `--teachers` | Teacher detectors used when `--teacher ensemble` is selected. Default: `paddleocr,ctd`. |
| `--ensemble-iou` | Minimum IoU used to treat two ensemble detections as duplicates. Default: `0.5`. |
| `--ensemble-overlap` | Minimum intersection over smaller area used to remove duplicate fragments. Default: `0.8`. |
| `--source` | Source dataset name under `dataset/` or an absolute path. Default: `manga-segment.v20i.yolo26`. |
| `--dest` | Destination dataset name under `dataset/` or an absolute path. Default: `manga-segment-text.v20i.yolo`. |
| `--class-name` | Name of the new class added to the dataset. Default: `text`. |
| `--text-only` | Generate a single-class dataset containing only text labels instead of merging with the original labels. |
| `--class-id` | Class id used with `--text-only`. Default: `0`. Ignored when labels are merged; the id is derived from the destination classes. |
| `--lang` | PaddleOCR language code, for example `en`, `japan`, or `ml`. Default: `en`. |
| `--device` | Inference device for the teacher. Choices: `auto`, `gpu`, or `cpu`. Default: `auto`. |
| `--det-limit-side-len` | Maximum side length used by the PaddleOCR detector. Default: `1536`. |
| `--det-db-unclip-ratio` | PaddleOCR DB contour expansion factor. Default: `1.6`. |
| `--det-db-thresh` | PaddleOCR DB binarization threshold. Lower values detect weaker or rotated text but can add noise. Default: `0.3`. |
| `--det-db-box-thresh` | PaddleOCR box confidence threshold. Lower values increase recall but can add false positives. Default: `0.5`. |
| `--tta-angles` | Comma-separated rotation test-time augmentation angles. Default: `0,90,180,270`. |
| `--no-rotation-tta` | Disable rotation test-time augmentation and detect only at `0` degrees. |
| `--merge-iou` | IoU threshold used to merge duplicate detections from different TTA angles. Default: `0.5`. |
| `--ctd-model-path` | Path to a CTD ONNX model. If omitted, the model is downloaded or loaded from `models/ctd/`. |
| `--ctd-model-url` | Alternative URL used to download the CTD ONNX model when the file does not exist. |
| `--ctd-input-size` | Square input size for the CTD ONNX model. Default: `1024`. |
| `--ctd-polygon-mode` | Method for converting CTD detections to polygons. Choices: `mask` or `line_box`. Default: `mask`. |
| `--ctd-mask-thresh` | CTD mask or DB map threshold before contour extraction. Default: `0.3`. |
| `--ctd-box-thresh` | CTD polygon confidence threshold when using `--ctd-polygon-mode line_box`. Default: `0.4`. |
| `--ctd-unclip-ratio` | DB polygon expansion factor when using `--ctd-polygon-mode line_box`. Default: `1.5`. |
| `--ctd-contour-epsilon-ratio` | CTD mask contour simplification ratio. Lower values preserve more points. Default: `0.002`. |
| `--ctd-max-candidates` | Maximum CTD contours evaluated per image. Default: `1000`. |
| `--min-polygon-area` | Minimum polygon area in pixels. Default: `10.0`. |
| `--min-polygon-short-side` | Minimum short side of the rotated text rectangle in pixels. Default: `3.0`. |
| `--min-polygon-long-side` | Minimum long side of the rotated text rectangle in pixels. Default: `8.0`. |
| `--tiny-polygon-short-side` | Short-side threshold below which the extreme aspect-ratio filter is applied. Default: `6.0`. |
| `--max-tiny-polygon-aspect-ratio` | Maximum aspect ratio for very thin polygons. Use `0` to disable this filter. Default: `40.0`. |
| `--precision` | Decimal places used for normalized label coordinates. Default: `6`. |
| `--overwrite` | Reprocess images even when an output label already exists. Without this flag, the command is resumable. |
| `--no-preview` | Do not save visual preview images. |
| `--preview-dir` | Directory for preview images. Default: `<dest>/previews`. |
| `--show` | Open a live window with each annotated page. Requires a display. |
| `--log-file` | Log file path. Default: `logs/text_masks_<timestamp>.log`. |
| `--log-level` | File log level. Default: `INFO`. |

## `class-videos`

Extracts centered object crops from a YOLO segmentation dataset and generates one video per class with `ffmpeg`.

```bash
uv run python src/main.py class-videos --dataset manga-segment_v2-v7-yolo26 --fps 8
```

The same command is also available as a standalone script:

```bash
uv run python src/dataset/class_videos.py --dataset manga-segment_v2-v7-yolo26 --fps 8
```

| Parameter | Description |
| --- | --- |
| `--dataset` | YOLO dataset folder containing `data.yaml` and split directories. Accepts an absolute path or a name relative to `dataset/`. Default: `manga-segment_v2-v7-yolo26`. |
| `--output` | Output directory for generated frames and videos. Default: `previews/class_videos`. |
| `--count` | Maximum number of instances per class. Default: all dataset instances. |
| `--canvas` | Square canvas size in pixels for each frame. Default: `640`. |
| `--fps` | Frames per second for each generated video. Default: `5.0`. |
| `--margin` | Extra context around each object, as a fraction of the largest object side. Default: `0.1`. Larger values keep more surrounding page context. |
| `--bg` | Background color used with `--mask` or when crops extend past page edges. Choices: `black`, `gray`, `white`. Default: `black`. |
| `--mask` | Fill everything outside the object polygon with the background color, isolating the object. |
| `--classes` | Optional subset of class names to process. Default: all classes in `data.yaml`. |
| `--splits` | Dataset splits to scan for instances. Default: `train valid test`. |
| `--seed` | Random seed used to shuffle frame order reproducibly. Default: `0`. |
| `--keep-frames` | Keep intermediate `.jpg` frames on disk. By default, frames are deleted after each video is generated. |

## Standalone U-Net Preprocessing CLI

The U-Net preprocessing script converts the legacy `dados/train` tree into `dados_cache` for U-Net training.

```bash
uv run python src/algorithms/unet/preprocess.py --npz
```

| Parameter | Description |
| --- | --- |
| `--npz` | Resize images to `320x512` RGBA, add flipped augmentations, and save compressed Float32 `.npz` files under `dados_cache`. |
| `--png` | Resize images to `320x512` RGBA, add flipped augmentations, and save `.png` files under `dados_cache`. |
| `--verify` | Compare image and mask pairs and print pairs whose average grayscale difference is above the verification threshold. |
