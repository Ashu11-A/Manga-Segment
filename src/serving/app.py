"""HTTP proxy layer that segments or annotates remote manga images."""

from io import BytesIO

import cv2 as cv
import numpy as np
from flask import Flask, request, send_file, Response

from serving.download import download_image
from serving.format import format_url

def create_app(*, algo="yolo", model_id=None, model_dir=None, size=1280, default_mode="segmented") -> Flask:
	"""Build a Flask app whose ``/image`` endpoint proxies and processes images."""
	# Lazy imports: the framework (and its heavy deps) are only needed at app build time.
	import algorithms

	algorithms.load_all()
	from core.registry import get_algorithm

	algo_obj = get_algorithm(algo, size=size, model_id=model_id, model_dir=model_dir)
	ref = algo_obj.resolve_model_ref(model_id=model_id, model_dir=model_dir)
	segmenter = algo_obj.build_segmenter(ref)

	app = Flask(__name__)

	@app.route("/image")
	def image():
		"""Download the image at ``url`` and return it segmented or annotated."""
		image_url = request.args.get("url")
		if not image_url:
			return "bandwidth-hero-proxy"

		image_url = format_url(image_url)
		if not image_url:
			return Response("URL inválida!", mimetype="text/plain", status=400)

		mode = request.args.get("mode", default_mode)

		try:
			data = download_image(image_url)
		except Exception:
			return Response("Erro ao baixar a imagem.", mimetype="text/plain", status=400)

		arr = np.frombuffer(data, np.uint8)
		image_bgr = cv.imdecode(arr, cv.IMREAD_COLOR)

		if mode == "segmented":
			bgra = segmenter.segment_array(image_bgr)
			if bgra is None:
				return send_file(BytesIO(data), mimetype="image/png")
			ok, buf = cv.imencode(".png", bgra)
			return send_file(BytesIO(buf.tobytes()), mimetype="image/png")

		if mode == "annotated":
			if hasattr(segmenter, "annotate"):
				annotated = segmenter.annotate(image_bgr)
				ok, buf = cv.imencode(".png", annotated)
				return send_file(BytesIO(buf.tobytes()), mimetype="image/png")
			return Response(
				f"'annotated' mode is not supported by algorithm '{algo}'",
				mimetype="text/plain",
				status=400,
			)

		return Response(f"Unknown mode '{mode}'", mimetype="text/plain", status=400)

	return app

def run_server(*, host="127.0.0.1", port=5000, **create_app_kwargs) -> None:
	"""Create and run the serving app (blocking)."""
	app = create_app(**create_app_kwargs)
	app.run(host=host, port=port, debug=False, use_reloader=False)
