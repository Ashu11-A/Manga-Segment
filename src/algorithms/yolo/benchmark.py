"""Parallel multi-version YOLO training + comparative report generation.

This module powers the parallel (multi-model training) and report
(regenerate the comparison) capabilities exposed by the YOLO algorithm.

It trains a set of *small* segmentation variants from several YOLO families
side by side, then parses each run's ``results.csv`` / ``args.yaml`` to build a
dynamic Markdown report comparing their efficiency. The report mirrors the
"Comparison" table documented in the project README so the two stay aligned.
"""

from __future__ import annotations

import csv
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from glob import glob

import yaml

from core import paths
from core.device import cuda_device_count

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Smallest *segmentation* checkpoint available for each family. Families that do
# not ship an official "-seg" checkpoint (e.g. YOLOv10) are still listed: the
# worker will try to load them and gracefully report them as unavailable instead
# of aborting the whole benchmark.
DEFAULT_MODELS: dict[str, str] = {
	"YOLOv8": "yolov8s-seg.pt",
	"YOLOv9": "yolov9c-seg.pt",   # v9 ships seg only as compact (c) / extended (e)
	"YOLOv10": "yolov10s-seg.pt",  # no official seg checkpoint -> reported as N/A
	"YOLO11": "yolo11s-seg.pt",
	"YOLO12": "yolo12s-seg.pt",
	"YOLO26": "yolo26s-seg.pt",
}

# Reference architecture facts used as a fallback / annotation when the values
# cannot be read directly from the loaded torch graph.
# Filter progression = backbone stage widths.
# Source: https://github.com/ultralytics/ultralytics/issues/189
ARCH_REFERENCE: dict[str, dict] = {
	"YOLOv8": {"kernel": 3, "filters": [64, 128, 256, 512, 768]},
	"YOLOv9": {"kernel": 3, "filters": [64, 128, 256, 512, 512]},
	"YOLOv10": {"kernel": 3, "filters": [64, 128, 256, 512, 1024]},
	"YOLO11": {"kernel": 3, "filters": [64, 128, 256, 512, 1024]},
	"YOLO12": {"kernel": 3, "filters": [64, 128, 256, 512, 1024]},
	"YOLO26": {"kernel": 3, "filters": [64, 128, 256, 512, 1024]},
}

# Trained models are saved here (one sub-folder per family), NOT in the
# ephemeral, git-ignored ``runs/``. Each sub-folder holds the *complete*
# training output (weights, args.yaml, results.csv, plots, curves) — the same
# layout as ``models/yolo`` — so models persist for inference (``--model-dir``)
# and future review.
PROJECT_DIR = paths.models_dir("yolo")
DEFAULT_DATA_PATH = paths.dataset_path("yolo", "data.yaml")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "best_hyperparameters.yaml")


# ---------------------------------------------------------------------------
# Worker (runs in its own process so each training is fully isolated)
# ---------------------------------------------------------------------------
def _train_model_worker(config: dict) -> dict:
	"""Train a single model in an isolated process and return its run metadata.

	Must stay at module level (picklable) so it works with the 'spawn' start
	method required by CUDA. Any failure is captured and returned instead of
	raising, so one broken family never takes down the whole benchmark.
	"""
	label = config["label"]
	model_name = config["model_name"]

	# Pin the GPU *before* importing torch/ultralytics so each worker only sees
	# the device assigned to it (one model per GPU when several are available).
	device_index = config.get("device_index")
	if device_index is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

	result = {
		"label": label,
		"model_name": model_name,
		"run_dir": os.path.join(config["project"], config["name"]),
		"status": "failed",
		"error": None,
		"onnx_exported": False,
	}

	try:
		import torch
		from ultralytics import YOLO

		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = YOLO(model_name, task="segment").to(device)

		model.train(
			data=config["data_path"],
			cfg=config["config_path"],
			project=config["project"],
			name=config["name"],
			exist_ok=True,
			patience=config["patience"],
			epochs=config["epochs"],
			batch=config["batch"],
			imgsz=config["size"],
			cache=True,
			optimizer="MuSGD",
			rect=True,
			multi_scale=0.25,
			mask_ratio=2,
			dropout=config["dropout"],
			val=True,
			plots=True,
			save=True,
			save_period=50,
		)

		result["status"] = "completed"

		# Export ONNX alongside the .pt so the saved model mirrors models/yolo/*
		# and is ready for inference. A failed export must not fail the run.
		try:
			model.export(format="onnx", imgsz=config["size"], half=(device == "cuda"))
			result["onnx_exported"] = True
		except Exception as export_exc:  # noqa: BLE001
			print(f"⚠️ [{label}] ONNX export skipped: {export_exc}")
	except Exception as exc:  # noqa: BLE001 - we deliberately capture everything
		result["error"] = f"{type(exc).__name__}: {exc}"
		result["trace"] = traceback.format_exc()
		print(f"❌ [{label}] training failed: {result['error']}")

	return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_benchmark(
	models: dict[str, str] | None = None,
	max_workers: int | None = None,
	size: int | list[int] = 1280,
	data_path: str = DEFAULT_DATA_PATH,
	config_path: str = DEFAULT_CONFIG_PATH,
	epochs: int = 1000,
	patience: int = 100,
	batch: int = 5,
	dropout: float = 0.1,
	project: str = PROJECT_DIR,
) -> str:
	"""Train several YOLO families in parallel, then emit the comparison report.

	``max_workers`` controls how many trainings run *simultaneously*. It defaults
	to the number of visible CUDA devices (one model per GPU) or 1 on CPU/single
	GPU. Forcing it above the GPU count shares VRAM across runs and may OOM.
	"""
	models = models or DEFAULT_MODELS

	gpu_count = cuda_device_count()
	if max_workers is None:
		max_workers = max(1, gpu_count)
	max_workers = min(max_workers, len(models))

	print("=" * 70)
	print("🚀 Parallel YOLO benchmark")
	print(f"   Models      : {', '.join(models)}")
	print(f"   Concurrency : {max_workers} worker(s)  |  GPUs detected: {gpu_count}")
	print(f"   Image size  : {size}  |  Epochs: {epochs}  |  Patience: {patience}")
	print(f"   Saving to   : {project}/<FAMILY>  (complete training data, persisted)")
	if max_workers > max(1, gpu_count):
		print("   ⚠️  Workers exceed available GPUs — runs will share VRAM (OOM risk).")
	print("=" * 70)

	configs = []
	for i, (label, model_name) in enumerate(models.items()):
		configs.append({
			"label": label,
			"model_name": model_name,
			"data_path": data_path,
			"config_path": config_path,
			"project": project,
			"name": label,
			"size": size,
			"epochs": epochs,
			"patience": patience,
			"batch": batch,
			"dropout": dropout,
			"device_index": (i % gpu_count) if gpu_count > 1 else None,
		})

	results: list[dict] = []
	if max_workers == 1:
		# Sequential: simplest and safest on a single GPU.
		for cfg in configs:
			print(f"\n▶️  Training {cfg['label']} ({cfg['model_name']})…")
			results.append(_train_model_worker(cfg))
	else:
		import multiprocessing as mp

		ctx = mp.get_context("spawn")  # required for CUDA in child processes
		with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
			futures = {pool.submit(_train_model_worker, cfg): cfg["label"] for cfg in configs}
			for future in as_completed(futures):
				results.append(future.result())

	ok = [r for r in results if r["status"] == "completed"]
	print(f"\n✅ Finished: {len(ok)}/{len(results)} models trained successfully.")

	return generate_report(project=project, data_path=data_path, train_results=results)


# ---------------------------------------------------------------------------
# Metrics / architecture extraction
# ---------------------------------------------------------------------------
def _safe_float(value) -> float | None:
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _parse_results_csv(run_dir: str) -> list[dict]:
	path = os.path.join(run_dir, "results.csv")
	if not os.path.exists(path):
		return []
	rows: list[dict] = []
	with open(path, newline="") as handle:
		for raw in csv.DictReader(handle):
			rows.append({(k or "").strip(): v for k, v in raw.items()})
	return rows


def _best_row(rows: list[dict]) -> dict | None:
	"""Pick the epoch with the highest mask mAP50-95 (the checkpoint policy)."""
	if not rows:
		return None

	def fitness(row: dict) -> float:
		m95 = _safe_float(row.get("metrics/mAP50-95(M)")) or 0.0
		m50 = _safe_float(row.get("metrics/mAP50(M)")) or 0.0
		return 0.9 * m95 + 0.1 * m50

	return max(rows, key=fitness)


def _read_args(run_dir: str) -> dict:
	path = os.path.join(run_dir, "args.yaml")
	if not os.path.exists(path):
		return {}
	with open(path) as handle:
		return yaml.safe_load(handle) or {}


def _extract_architecture(run_dir: str, label: str) -> dict:
	"""Best-effort architecture facts read from the trained weights.

	Falls back to ``ARCH_REFERENCE`` when the graph cannot be inspected (e.g. the
	report is regenerated on a machine without the weights handy).
	"""
	ref = ARCH_REFERENCE.get(label, {})
	info = {
		"channels": 3,
		"kernel": ref.get("kernel", 3),
		"filters": ref.get("filters"),
		"filters_source": "reference",
	}

	weights = os.path.join(run_dir, "weights", "best.pt")
	if not os.path.exists(weights):
		weights = os.path.join(run_dir, "weights", "last.pt")
	if not os.path.exists(weights):
		return info

	try:
		import torch.nn as nn
		from ultralytics import YOLO

		net = YOLO(weights, task="segment").model
		yaml_cfg = getattr(net, "yaml", {}) or {}
		info["channels"] = int(yaml_cfg.get("ch", 3) or 3)

		kernels: set[int] = set()
		filters: list[int] = []
		for module in net.modules():
			if isinstance(module, nn.Conv2d):
				k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
				if k > 1:
					kernels.add(int(k))
				stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
				# Stride-2 convs mark backbone stage transitions -> width progression.
				if stride == 2:
					filters.append(int(module.out_channels))

		if kernels:
			info["kernel"] = max(kernels)  # dominant spatial kernel
		if filters:
			info["filters"] = filters
			info["filters_source"] = "detected"
	except Exception as exc:  # noqa: BLE001
		print(f"⚠️  Could not inspect architecture for {label}: {exc}")

	return info


def _dataset_info(data_path: str) -> dict:
	info = {"images": None, "train": None, "val": None, "test": None, "classes": None}
	if not os.path.exists(data_path):
		return info
	try:
		with open(data_path) as handle:
			data = yaml.safe_load(handle) or {}
	except Exception:
		return info

	names = data.get("names")
	if isinstance(names, dict):
		info["classes"] = list(names.values())
	elif isinstance(names, list):
		info["classes"] = names

	base = data.get("path") or os.path.dirname(data_path)
	if not os.path.isabs(base):
		base = os.path.normpath(os.path.join(os.path.dirname(data_path), base))

	total = 0
	for split in ("train", "val", "test"):
		rel = data.get(split)
		if not rel:
			continue
		count = _count_images(base, rel)
		info[split] = count
		if count:
			total += count
	info["images"] = total or None
	return info


def _count_images(base: str, rel: str) -> int | None:
	candidate = rel if os.path.isabs(rel) else os.path.join(base, rel)
	# data.yaml usually points at the images folder for the split.
	if os.path.isdir(candidate):
		exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
		return sum(len(glob(os.path.join(candidate, "**", e), recursive=True)) for e in exts)
	return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _collect_model_record(label: str, run_dir: str, train_result: dict | None) -> dict:
	rows = _parse_results_csv(run_dir)
	best = _best_row(rows)
	args = _read_args(run_dir)
	arch = _extract_architecture(run_dir, label)

	has_pt = os.path.isfile(os.path.join(run_dir, "weights", "best.pt"))
	has_onnx = bool(glob(os.path.join(run_dir, "weights", "*.onnx")))

	record = {
		"label": label,
		"run_dir": run_dir,
		"model_name": (train_result or {}).get("model_name", DEFAULT_MODELS.get(label, "?")),
		"status": (train_result or {}).get("status", "completed" if rows else "missing"),
		"error": (train_result or {}).get("error"),
		"has_weights": has_pt,
		"has_onnx": has_onnx,
		"precision": None,
		"recall": None,
		"seg_loss": None,
		"cls_loss": None,
		"map5095": None,
		"map50": None,
		"stop_epoch": None,
		"max_epochs": args.get("epochs"),
		"early_stopping": args.get("patience"),
		"early_triggered": None,
		"imgsz": args.get("imgsz"),
		"dropout": args.get("dropout"),
		"channels": arch["channels"],
		"kernel": arch["kernel"],
		"filters": arch["filters"],
		"filters_source": arch["filters_source"],
	}

	if best:
		record["precision"] = _safe_float(best.get("metrics/precision(M)"))
		record["recall"] = _safe_float(best.get("metrics/recall(M)"))
		record["seg_loss"] = _safe_float(best.get("val/seg_loss"))
		record["cls_loss"] = _safe_float(best.get("val/cls_loss"))
		record["map5095"] = _safe_float(best.get("metrics/mAP50-95(M)"))
		record["map50"] = _safe_float(best.get("metrics/mAP50(M)"))

	if rows:
		last_epoch = _safe_float(rows[-1].get("epoch"))
		record["stop_epoch"] = int(last_epoch) if last_epoch is not None else len(rows)
		if record["max_epochs"]:
			record["early_triggered"] = record["stop_epoch"] < int(record["max_epochs"]) - 1

	return record


def _fmt(value, digits: int = 5, dash: str = "❌") -> str:
	if value is None:
		return dash
	if isinstance(value, float):
		return f"{value:.{digits}f}"
	return str(value)


def _fmt_imgsz(value) -> str:
	if value is None:
		return "❌"
	if isinstance(value, list):
		return "x".join(str(v) for v in value)
	return f"{value}x{value}"


def _discover_runs(models_dir: str) -> dict[str, str]:
	"""Map ``family -> run dir`` for every trained model under ``models_dir``.

	A sub-folder counts as a trained run when it holds a ``results.csv`` or a
	``weights/best.pt``. This naturally skips bare ``weights`` folders and the
	top-level files of a flat model, keying each run by its folder name.
	"""
	found: dict[str, str] = {}
	for path in sorted(glob(os.path.join(models_dir, "*"))):
		if not os.path.isdir(path):
			continue
		if os.path.exists(os.path.join(path, "results.csv")) or os.path.exists(
			os.path.join(path, "weights", "best.pt")
		):
			found[os.path.basename(path)] = path
	return found


def generate_report(
	project: str = PROJECT_DIR,
	data_path: str = DEFAULT_DATA_PATH,
	train_results: list[dict] | None = None,
	output_path: str | None = None,
) -> str:
	"""Build the comparative Markdown report from benchmark run directories."""
	results_by_label = {r["label"]: r for r in (train_results or [])}

	# Discover run directories: prefer those we just trained, else auto-discover
	# every saved family sub-folder under the models directory.
	run_dirs: dict[str, str] = {}
	if train_results:
		for r in train_results:
			run_dirs[r["label"]] = r["run_dir"]
	else:
		run_dirs = _discover_runs(project)

	if not run_dirs:
		raise ValueError(
			f"No trained models found in {project}. Run training with --parallel first."
		)

	# Present families in the canonical registry order; unknown labels go last.
	order = list(DEFAULT_MODELS)
	run_dirs = dict(
		sorted(run_dirs.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else len(order), kv[0]))
	)

	records = [
		_collect_model_record(label, run_dir, results_by_label.get(label))
		for label, run_dir in run_dirs.items()
	]

	trained = [r for r in records if r["map5095"] is not None]
	best = max(trained, key=lambda r: r["map5095"]) if trained else None
	dataset = _dataset_info(data_path)

	md = _render_markdown(records, best, dataset)

	output_path = output_path or os.path.abspath(os.path.join(project, "..", "..", "BENCHMARK_REPORT.md"))
	with open(output_path, "w") as handle:
		handle.write(md)

	print(f"\n📄 Report written to: {output_path}")
	if best:
		print(f"🏆 Best model: {best['label']} (mask mAP50-95 = {_fmt(best['map5095'])})")
	return output_path


def _render_markdown(records: list[dict], best: dict | None, dataset: dict) -> str:
	now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
	lines: list[str] = []

	lines.append("# 🧪 YOLO Segmentation Benchmark Report")
	lines.append("")
	lines.append(f"_Generated: {now}_")
	lines.append("")
	lines.append(
		"Comparative evaluation of small **segmentation** variants across YOLO "
		"families, trained on the Manga-Segment dataset for manga background / "
		"speech-balloon segmentation. Metrics are reported for the best checkpoint "
		"(highest mask `mAP50-95`) of each run."
	)
	lines.append("")

	# --- Dataset ----------------------------------------------------------
	lines.append("## 📚 Dataset")
	lines.append("")
	if dataset.get("classes"):
		lines.append(f"- **Classes ({len(dataset['classes'])}):** {', '.join(map(str, dataset['classes']))}")
	for key, title in (("images", "Total images"), ("train", "Train"), ("val", "Valid"), ("test", "Test")):
		if dataset.get(key) is not None:
			lines.append(f"- **{title}:** {dataset[key]}")
	lines.append("")

	# --- Verdict ----------------------------------------------------------
	lines.append("## 🏆 Best Model")
	lines.append("")
	if best:
		lines.append(
			f"**{best['label']}** (`{best['model_name']}`) achieved the highest mask "
			f"segmentation quality with **mAP50-95 = {_fmt(best['map5095'])}** "
			f"(mAP50 = {_fmt(best['map50'])})."
		)
		lines.append("")
		lines.append("| Metric | Value |")
		lines.append("|--|--|")
		lines.append(f"| Precision (M) | {_fmt(best['precision'])} |")
		lines.append(f"| Recall (M) | {_fmt(best['recall'])} |")
		lines.append(f"| Val Seg Loss | {_fmt(best['seg_loss'])} |")
		lines.append(f"| Val Class Loss | {_fmt(best['cls_loss'])} |")
		lines.append(f"| Early Stopping (patience) | {_fmt(best['early_stopping'], 0)} |")
		lines.append(f"| Stop Epoch | {_fmt(best['stop_epoch'], 0)} |")
		lines.append(f"| Training Size | {_fmt_imgsz(best['imgsz'])} |")
		lines.append(f"| Image Channels | {_fmt(best['channels'], 0)} |")
		lines.append(f"| Dropout | {_dropout_str(best['dropout'])} |")
		lines.append(f"| Kernel Size | {_fmt(best['kernel'], 0)} |")
		lines.append(f"| Filter | {_filters_str(best['filters'])} |")
	else:
		lines.append("_No completed runs with metrics were found._")
	lines.append("")

	# --- Comparison table -------------------------------------------------
	lines.append("## 📊 Comparison")
	lines.append("")
	lines.append(
		"Mirrors the README *Comparison* table. `Image Set` is the dataset size; "
		"`Training Size` is the input resolution used for training."
	)
	lines.append("")

	header = ["Property"] + [r["label"] for r in records]
	lines.append("| " + " | ".join(header) + " |")
	lines.append("|" + "|".join(["--"] * len(header)) + "|")

	img_set = dataset.get("images")
	row_specs = [
		("Precision", lambda r: _fmt(r["precision"])),
		("Recall", lambda r: _fmt(r["recall"])),
		("Val Seg Loss", lambda r: _fmt(r["seg_loss"])),
		("Val Class Loss", lambda r: _fmt(r["cls_loss"])),
		("mAP50 (M)", lambda r: _fmt(r["map50"])),
		("mAP50-95 (M)", lambda r: _fmt(r["map5095"])),
		("Pretrained Model", lambda r: f"`{r['model_name']}`"),
		("Early Stopping", lambda r: _fmt(r["early_stopping"], 0)),
		("Stop Epoch", lambda r: _fmt(r["stop_epoch"], 0)),
		("Image Set", lambda r: str(img_set) if img_set is not None else "❌"),
		("Image Channels", lambda r: _fmt(r["channels"], 0)),
		("Training Size", lambda r: _fmt_imgsz(r["imgsz"])),
		("Dropout", lambda r: _dropout_str(r["dropout"])),
		("Kernel Size", lambda r: _fmt(r["kernel"], 0)),
		("Filter", lambda r: _filters_str(r["filters"])),
		("Status", lambda r: _status_str(r)),
	]
	for title, getter in row_specs:
		lines.append("| " + title + " | " + " | ".join(getter(r) for r in records) + " |")
	lines.append("")

	# --- Per-model detail -------------------------------------------------
	lines.append("## 🔍 Per-Model Details")
	lines.append("")
	for r in records:
		lines.append(f"### {r['label']} — `{r['model_name']}`")
		lines.append("")
		if r["status"] != "completed" and r["map5095"] is None:
			reason = r.get("error") or "no metrics found (segmentation checkpoint may be unavailable for this family)"
			lines.append(f"> ⚠️ **Not evaluated:** {reason}")
			lines.append("")
			continue
		lines.append(f"- **Run dir:** `{r['run_dir']}`")
		lines.append(f"- **Best mask mAP50-95 / mAP50:** {_fmt(r['map5095'])} / {_fmt(r['map50'])}")
		lines.append(f"- **Precision / Recall (M):** {_fmt(r['precision'])} / {_fmt(r['recall'])}")
		lines.append(f"- **Val seg / cls loss:** {_fmt(r['seg_loss'])} / {_fmt(r['cls_loss'])}")
		early = "triggered" if r["early_triggered"] else "not triggered"
		lines.append(
			f"- **Early stopping:** patience {_fmt(r['early_stopping'], 0)}, "
			f"stopped at epoch {_fmt(r['stop_epoch'], 0)} ({early})"
		)
		lines.append(
			f"- **Architecture:** kernel {r['kernel']}, channels {r['channels']}, "
			f"filters {_filters_str(r['filters'])} _({r['filters_source']})_"
		)
		artifacts = []
		if r["has_weights"]:
			artifacts.append("`weights/best.pt`")
		if r["has_onnx"]:
			artifacts.append("ONNX")
		lines.append(f"- **Saved artifacts:** {', '.join(artifacts) if artifacts else '❌ none'}")
		if r["has_weights"]:
			lines.append(f"- **Run inference:** `python main.py --test --model-dir {r['run_dir']}`")
		lines.append("")

	# --- Methodology ------------------------------------------------------
	lines.append("## 🧭 Methodology & Notes")
	lines.append("")
	lines.append(
		"- Each family is trained in an **isolated process** so a failure (or a "
		"missing segmentation checkpoint, e.g. YOLOv10) never aborts the rest."
	)
	lines.append(
		"- **Best epoch** is the one maximizing mask `mAP50-95` (Ultralytics' "
		"`best.pt` checkpoint policy)."
	)
	lines.append(
		"- **Filter** values are read from the trained graph when available "
		"(`detected`), otherwise from documented references (`reference`)."
	)
	lines.append(
		"- Filter reference: "
		"[ultralytics/ultralytics#189](https://github.com/ultralytics/ultralytics/issues/189)."
	)
	lines.append("")

	return "\n".join(lines)


def _dropout_str(value) -> str:
	if value in (None, "", 0, 0.0):
		return "❌"
	return str(value)


def _filters_str(filters) -> str:
	if not filters:
		return "❌"
	return "[" + ",".join(str(f) for f in filters) + "]"


def _status_str(record: dict) -> str:
	if record["status"] == "completed" or record["map5095"] is not None:
		return "✅"
	if record["status"] == "missing":
		return "➖ N/A"
	return "❌ failed"
