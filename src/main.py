"""Unified command-line interface for the Manga-Segment CV framework.

One subcommand CLI drives every registered algorithm (``--algo``):

    main.py train     --algo yolo  --size 1280
    main.py test      --algo unet  --model-dir ../models/unet
    main.py tune      --algo yolo
    main.py convert   --algo yolo  --model 1
    main.py benchmark --algo yolo  --workers 2
    main.py report    --algo yolo
    main.py serve     --algo yolo  --port 5000

Trained-model selection is unified across algorithms: ``--model-dir`` points at
an explicit model, ``--model`` selects a numbered run/model, and omitting both
falls back to each backend's default (latest YOLO run / ``models/unet``).
Capabilities a backend does not implement raise a friendly message rather than a
traceback.
"""

from __future__ import annotations

import argparse
import sys

from algorithms.yolo.models import ALL_MODELS as _YOLO_MODELS, DEFAULT_MODEL as _YOLO_DEFAULT


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="main.py",
		description="Manga-Segment: unified training / inference / serving CLI.",
	)
	sub = parser.add_subparsers(dest="command", required=True)

	def add_algo(sp: argparse.ArgumentParser) -> None:
		sp.add_argument(
			"--algo", default="yolo", help="Algorithm to use (e.g. yolo, unet)."
		)
		sp.add_argument(
			"--size",
			type=int,
			default=None,
			help="Input image size (default: auto-detect from the dataset images).",
		)

	def add_model_selection(sp: argparse.ArgumentParser) -> None:
		sp.add_argument("--model", type=int, default=None, help="Trained model id.")
		sp.add_argument(
			"--model-dir", default=None, help="Explicit trained-model path."
		)

	# train
	sp = sub.add_parser(
		"train",
		help="Train from scratch, fine-tune (--model), or resume (--resume).",
	)
	add_algo(sp)
	sp.add_argument("--model", type=int, default=None, help="Resume/base model id.")
	sp.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
	sp.add_argument("--batch", type=int, default=None, help="Batch size.")
	sp.add_argument(
		"--patience", type=int, default=None, help="Early-stopping patience."
	)
	sp.add_argument(
		"--resume",
		action="store_true",
		help=(
			"Resume an interrupted run from its last.pt, continuing toward the "
			"original epoch target. Use --model <id> to pick which run to resume "
			"(default: the latest); pass --batch to lower VRAM use after an OOM."
		),
	)
	sp.add_argument(
		"--base-model",
		default=_YOLO_DEFAULT,
		choices=_YOLO_MODELS,
		metavar="MODEL",
		help=(
			f"Base YOLO model to train from scratch (default: {_YOLO_DEFAULT}). "
			"Interactive wizard picks family + size when omitted in the wizard."
		),
	)

	# test (inference)
	sp = sub.add_parser("test", help="Run segmentation inference over a folder.")
	add_algo(sp)
	add_model_selection(sp)
	sp.add_argument(
		"--images", default=None, help="Input images dir (default: ../images)."
	)
	sp.add_argument("--output", default=None, help="Output dir (default: ../output).")
	sp.add_argument(
		"--threshold", type=float, default=0.5, help="U-Net mask threshold."
	)
	sp.add_argument(
		"--conf", type=float, default=0.5, help="YOLO confidence threshold."
	)
	sp.add_argument(
		"--keep-classes",
		nargs="*",
		default=None,
		help="Classes to keep in the composite.",
	)

	# tune
	sp = sub.add_parser("tune", help="Hyperparameter tuning.")
	add_algo(sp)
	sp.add_argument(
		"--base-model",
		default=_YOLO_DEFAULT,
		choices=_YOLO_MODELS,
		metavar="MODEL",
		help=f"Base YOLO model to tune (default: {_YOLO_DEFAULT}).",
	)

	# convert
	sp = sub.add_parser("convert", help="Export a trained model (ONNX/TFJS).")
	add_algo(sp)
	add_model_selection(sp)

	# benchmark
	sp = sub.add_parser(
		"benchmark", help="Train several model families in parallel + report."
	)
	add_algo(sp)
	sp.add_argument(
		"--workers", type=int, default=None, help="Parallel trainings (default: #GPUs)."
	)
	sp.add_argument("--epochs", type=int, default=1000)
	sp.add_argument("--patience", type=int, default=100)
	sp.add_argument("--batch", type=int, default=5)

	# report
	sp = sub.add_parser("report", help="Regenerate the comparative report.")
	add_algo(sp)

	# serve
	sp = sub.add_parser("serve", help="Start the HTTP segmentation proxy.")
	add_algo(sp)
	add_model_selection(sp)
	sp.add_argument("--host", default="127.0.0.1")
	sp.add_argument("--port", type=int, default=5000)
	sp.add_argument(
		"--mode",
		choices=["segmented", "annotated"],
		default="segmented",
		help="Default output mode for the /image endpoint.",
	)

	# download (interactive wizard: pick dataset version + model, download, train)
	sp = sub.add_parser(
		"download",
		help="Interactive wizard: select dataset version and model, download from Roboflow, then train.",
	)
	from dataset.download import add_download_arguments

	add_download_arguments(sp)

	# distill (teacher model -> YOLO instance-segmentation dataset)
	sp = sub.add_parser(
		"distill",
		help="Distill a teacher (PaddleOCR) into a YOLO-seg text dataset for Roboflow.",
	)
	# Imported lazily: the module keeps heavy deps (cv2/paddle) inside functions,
	# so building the parser stays cheap.
	from dataset.distill import add_distill_arguments

	add_distill_arguments(sp)

	return parser


def main(argv: list[str] | None = None) -> int:
	raw = list(sys.argv[1:]) if argv is None else list(argv)

	# No arguments → launch the interactive arrow-key wizard. It assembles the
	# equivalent argv (printing the resulting command), which is then parsed and
	# dispatched exactly as if it had been typed on the command line.
	if not raw:
		from interactive import run_wizard

		raw = run_wizard(build_parser)
		if not raw:
			return 1

	args = build_parser().parse_args(raw)

	# download runs the interactive Roboflow wizard and optionally starts training.
	if args.command == "download":
		from dataset.download import run_download

		return run_download(args)

	# distill is a dataset-prep command: it does not touch the algorithm registry
	# (no torch/ultralytics import), so dispatch it before loading algorithms.
	# PaddleOCR conflicts with this project's tensorflow (protobuf pin), so it is
	# not installed here; run it via the isolated `src/dataset/run.py` script.
	if args.command == "distill":
		from dataset.distill import run_distill

		try:
			# PaddleOCR/opencv are imported lazily deep inside run_distill.
			return run_distill(args)
		except ImportError as missing_dependency:
			forwarded_args = argv if argv is not None else sys.argv[1:]
			print(
				"❌ PaddleOCR is not available in this environment "
				f"({missing_dependency}).\n"
				"   It conflicts with tensorflow, so run the isolated script instead:\n"
				"       uv run src/dataset/run.py " + " ".join(forwarded_args),
				file=sys.stderr,
			)
			return 1

	# Heavy imports happen only after argument parsing (so --help stays fast).
	import algorithms

	algorithms.load_all()
	from core import paths
	from core.algorithm import NotSupported
	from core.registry import available, get_algorithm

	# serve is wired through the serving layer, not a BaseAlgorithm method.
	if args.command == "serve":
		from serving.app import run_server

		run_server(
			algo=args.algo,
			model_id=args.model,
			model_dir=args.model_dir,
			size=args.size,
			default_mode=args.mode,
			host=args.host,
			port=args.port,
		)
		return 0

	if args.algo not in available():
		print(
			f"❌ Unknown algorithm '{args.algo}'. Available: {', '.join(available())}.",
			file=sys.stderr,
		)
		return 2

	extra: dict = {}
	if bm := getattr(args, "base_model", None):
		extra["model_name"] = bm

	algo = get_algorithm(
		args.algo,
		size=args.size,
		model_id=getattr(args, "model", None),
		model_dir=getattr(args, "model_dir", None),
		**extra,
	)

	try:
		if args.command == "train":
			kwargs = {
				key: getattr(args, key)
				for key in ("epochs", "batch", "patience")
				if getattr(args, key) is not None
			}
			if args.model is not None:
				kwargs["model_id"] = args.model
			if args.resume:
				kwargs["resume"] = True
			algo.train(**kwargs)

		elif args.command == "test":
			written = algo.test(
				images_dir=args.images or paths.images_dir(),
				output_dir=args.output or paths.output_dir(),
				model_id=args.model,
				model_dir=args.model_dir,
				keep_classes=args.keep_classes,
				threshold=args.threshold,
				conf=args.conf,
			)
			print(f"✅ {len(written)} file(s) written.")

		elif args.command == "tune":
			algo.tune()

		elif args.command == "convert":
			path = algo.convert(model_id=args.model, model_dir=args.model_dir)
			print(f"✅ Converted model at: {path}")

		elif args.command == "benchmark":
			report = algo.benchmark(
				max_workers=args.workers,
				epochs=args.epochs,
				patience=args.patience,
				batch=args.batch,
			)
			print(f"📄 Report: {report}")

		elif args.command == "report":
			report = algo.report()
			print(f"📄 Report: {report}")

	except NotSupported as exc:
		print(f"❌ {exc}", file=sys.stderr)
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
