"""U-Net algorithm: inference, conversion, training and tuning.

Registered backend exposing the U-Net capabilities through the framework's
:class:`~core.algorithm.BaseAlgorithm` interface. Inference is the fully
supported path (loads the pre-trained ``SavedModel``); training/tuning/convert
are faithful ports of the legacy Keras pipeline and require the dataset tree
plus the ``tensorflowjs`` / ``keras-tuner`` dependencies.

All heavy imports (tensorflow/keras/keras_tuner/tensorflowjs) are kept lazy
inside the methods so ``import algorithms`` and ``algorithms.load_all()`` stay
cheap and never pull in TensorFlow just to register this backend.
"""

from __future__ import annotations

import os
from typing import Any

from core import paths
from core.algorithm import BaseAlgorithm
from core.registry import register
from core.segmenter import BaseSegmenter


@register
class UnetAlgorithm(BaseAlgorithm):
	"""Pre-trained U-Net segmentation (and the legacy Keras training pipeline)."""

	name = "unet"
	display_name = "U-Net"

	# -- model selection + inference -----------------------------------------
	def resolve_model_ref(
		self,
		*,
		model_id: int | None = None,
		model_dir: str | None = None,
	) -> str:
		"""Resolve a SavedModel directory.

		An explicit ``model_dir`` wins; otherwise ``model_id`` selects
		``models/unet-<id>``; otherwise the default ``models/unet``.
		"""
		if model_dir is not None:
			return os.path.abspath(os.path.expanduser(model_dir))
		if model_id is not None:
			return paths.models_dir(f"unet-{model_id}")
		return paths.models_dir("unet")

	def build_segmenter(
		self,
		model_ref: str,
		*,
		threshold: float = 0.5,
		**kwargs: Any,
	) -> BaseSegmenter:
		"""Build a :class:`~algorithms.unet.segmenter.UnetSegmenter`."""
		from algorithms.unet.segmenter import UnetSegmenter

		return UnetSegmenter(model_ref, threshold=threshold)

	def test(self, *, save_masks: bool = False, **kwargs: Any) -> list[str]:
		"""Run inference, writing only ``<stem>_unet_mask.png`` + ``<stem>_segmented.png``.

		Overrides the default so per-instance grayscale masks are *not* written
		(the single ``foreground`` instance would otherwise add a redundant file),
		matching the legacy ``UnetModel.test`` output set.
		"""
		return super().test(save_masks=save_masks, **kwargs)

	# -- conversion ----------------------------------------------------------
	def convert(
		self,
		*,
		model_id: int | None = None,
		model_dir: str | None = None,
		**kwargs: Any,
	) -> str:
		"""Export a trained SavedModel to TensorFlow.js (ports ``unetConvert``)."""
		import tensorflowjs as tfjs

		saved_model = self.resolve_model_ref(model_id=model_id, model_dir=model_dir)
		if not os.path.isdir(saved_model):
			raise FileNotFoundError(f"⚠️ U-Net model not found: {saved_model}")

		# The trainer stores the servable graph under a ``best_model`` subdir when
		# present, falling back to the model directory itself.
		source = os.path.join(saved_model, "best_model")
		if not os.path.isdir(source):
			source = saved_model

		tfjs.converters.convert_tf_saved_model(
			saved_model_dir=source, output_dir=saved_model
		)
		print(f"✅ Converted U-Net to TFJS in: {saved_model}")
		return saved_model

	# -- training ------------------------------------------------------------
	def train(
		self,
		*,
		epochs: int = 250,
		batch_size: int = 32,
		model_id: int | None = None,
		data_dir: str = "dados_cache/train",
		**kwargs: Any,
	) -> Any:
		"""Train a U-Net (faithful port of the legacy ``unetTraining`` standard flow).

		Requires the cached dataset tree (``data_dir``) and the ``tensorflowjs``
		dependency. Builds the network with :func:`architecture.LoaderModel` (or
		resumes ``model_id``) and the ``DataLoader``/``TensorLoader`` pipeline.
		"""
		return self._run_training(
			epochs=epochs,
			batch_size=batch_size,
			model_id=model_id,
			data_dir=data_dir,
			search_best=False,
		)

	def tune(
		self,
		*,
		epochs: int = 250,
		data_dir: str = "dados_cache/train",
		**kwargs: Any,
	) -> Any:
		"""Hyperparameter search via keras_tuner Hyperband (legacy ``--best`` path).

		Requires ``keras-tuner`` and the cached dataset tree.
		"""
		return self._run_training(
			epochs=epochs,
			batch_size=32,
			model_id=None,
			data_dir=data_dir,
			search_best=True,
		)

	# -- shared training implementation --------------------------------------
	def _run_training(
		self,
		*,
		epochs: int,
		batch_size: int,
		model_id: int | None,
		data_dir: str,
		search_best: bool,
	) -> Any:
		"""Build the data pipeline, fit (or search) and persist the model + report.

		Faithful consolidation of the legacy ``unetTraining``: the standard branch
		and the keras_tuner ``--best`` branch share one data/callback/saving path.
		"""
		import asyncio
		import json
		from datetime import datetime

		import keras
		import matplotlib.pyplot as plt
		import tensorflow as tf
		from keras import backend as K
		from keras.callbacks import (
			EarlyStopping,
			ModelCheckpoint,
			ReduceLROnPlateau,
			TensorBoard,
			TerminateOnNaN,
		)

		from algorithms.unet.architecture import FindModel, LoaderModel
		from algorithms.unet.data import DataLoader, TensorLoader

		# Mixed precision + soft device placement, as in the legacy trainer.
		gpus = tf.config.list_physical_devices("GPU")
		if gpus:
			try:
				tf.config.set_logical_device_configuration(
					gpus[0],
					[tf.config.LogicalDeviceConfiguration(memory_limit=6144)],
				)
			except RuntimeError as exc:
				print(exc)
		tf.random.set_seed(666)
		keras.mixed_precision.set_global_policy("mixed_float16")
		tf.config.set_soft_device_placement(True)

		logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

		loader_files = DataLoader()
		loader_tensor = TensorLoader()

		images, masks = asyncio.run(loader_files.LoadFiles(data_dir, onlyPath=True)) or (
			[],
			[],
		)
		if not images or not masks:
			print("Nenhum dado carregado!")
			return None

		total_model = self._count_models("models")
		out_dir = f"models/my-model-{total_model}"

		dataset = tf.data.Dataset.from_tensor_slices(
			(tf.constant(images), tf.constant(masks))
		).map(loader_tensor.convertImages)

		buffer_size = len(images)
		n_train = int(0.7 * buffer_size)
		n_validation = int(0.3 * buffer_size)

		dataset = dataset.cache().shuffle(buffer_size).repeat()
		validate_ds = (
			dataset.take(n_validation)
			.batch(batch_size)
			.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		)
		train_ds = (
			dataset.skip(n_validation)
			.take(n_train)
			.batch(batch_size)
			.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		)
		print(f"Treinamento: {n_train} | Validadores: {n_validation}")

		if search_best:
			import keras_tuner as kt

			print("Iniciando procura do melhor modelo!")
			tuner = kt.Hyperband(
				FindModel,
				objective="val_accuracy",
				max_epochs=epochs,
				max_consecutive_failed_trials=3,
				directory="models",
				project_name=f"my-model-{total_model}",
			)
			tuner.search(
				train_ds,
				validation_data=validate_ds,
				callbacks=[
					EarlyStopping(monitor="val_accuracy", patience=25, verbose=1),
					TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=2),
					TerminateOnNaN(),
					ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
				],
				batch_size=1,
			)
			best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
			if tuner.hypermodel is None:
				print("tuner.hypermodel é None -_-")
				return None
			model = tuner.hypermodel.build(best_hps)
		elif model_id is not None:
			print(f"Retreinando o Modelo: {model_id}")
			model = keras.models.load_model(f"models/my-model-{model_id}")
		else:
			model = LoaderModel()

		K.clear_session()
		os.makedirs(out_dir, exist_ok=True)

		history = model.fit(
			train_ds,
			validation_data=validate_ds,
			epochs=epochs,
			batch_size=1,
			steps_per_epoch=max(1, n_train // batch_size),
			validation_steps=max(1, n_validation // batch_size),
			callbacks=[
				ModelCheckpoint(
					f"{out_dir}/best_model",
					monitor="val_accuracy",
					save_best_only=True,
					mode="auto",
					verbose=1,
				),
				EarlyStopping(monitor="val_accuracy", patience=20, verbose=1),
				TensorBoard(log_dir=logs, histogram_freq=1),
				TerminateOnNaN(),
				ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
			],
		)

		keras.models.save_model(model, filepath=out_dir, overwrite=True)
		self._export_tfjs(out_dir)
		self._save_history_plot(history, out_dir)

		with open(f"{out_dir}/data.json", "w") as handle:
			json.dump(
				{
					"epochs": history.epoch,
					"history": history.history,
					"params": history.params,
				},
				handle,
			)

		print(f"Modelo salvo: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')}")
		return out_dir

	# -- training helpers ----------------------------------------------------
	@staticmethod
	def _count_models(directory: str) -> int:
		from glob import glob

		return len(glob(os.path.join(directory, "my-model-*")))

	@staticmethod
	def _export_tfjs(out_dir: str) -> None:
		try:
			import tensorflowjs as tfjs

			tfjs.converters.convert_tf_saved_model(
				saved_model_dir=f"{out_dir}/best_model", output_dir=out_dir
			)
		except Exception as exc:  # noqa: BLE001 - export must not fail training
			print(f"⚠️ TFJS export skipped: {exc}")

	@staticmethod
	def _save_history_plot(history: Any, out_dir: str) -> None:
		import matplotlib.pyplot as plt

		loss = history.history.get("loss", [])
		val_loss = history.history.get("val_loss", [])
		plt.figure()
		plt.plot(history.epoch, loss, "r", label="Training loss")
		plt.plot(history.epoch, val_loss, "b", label="Validation loss")
		plt.title("Training and Validation Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss Value")
		plt.ylim([0, 1])
		plt.legend()
		plt.savefig(f"{out_dir}/loss.png")
		plt.close()
