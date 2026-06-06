"""Dataset loading helpers for U-Net training.

``DataLoader`` walks the ``train``/``validation`` dataset tree (async, ported
verbatim from the legacy ``functions/getData.py``) and ``TensorLoader`` decodes
image/mask pairs into tensors (ported from ``tensor.py``). Both are only used by
the training/tuning code paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import tensorflow as tf
from tqdm import tqdm


class DataLoader:
	async def LoadFiles(
		self, markDir: str = "", onlyPath: bool = False
	) -> tuple[list[bytes], list[bytes]] | tuple[list[str], list[str]] | None:
		images: list[bytes] = []
		masks: list[bytes] = []

		imagesPath: list[str] = []
		masksPath: list[str] = []

		def scan_directory(directory: str):
			for entry in os.scandir(directory):
				if entry.is_dir(follow_symlinks=False):
					scan_directory(entry.path)
				elif entry.is_file(follow_symlinks=False) and entry.path.endswith(
					(".npz", ".png", ".jpg", ".jpeg", ".webp")
				):
					mark_path = Path(entry.path.replace("train", "validation"))
					if mark_path.exists():
						if onlyPath != True:
							with open(entry.path, "rb") as file:
								image_buffer = file.read()
								images.append(image_buffer)
							with open(mark_path, "rb") as file:
								mask_buffer = file.read()
								masks.append(mask_buffer)
						else:
							imagesPath.append(str(entry.path))
							masksPath.append(str(mark_path))
					else:
						print(f"Faltando: {entry.path.replace('train', 'validation')}")
				else:
					print(f"File Error: {entry.path}")

		if markDir == "":
			print("Nenhum diretório foi repassado!")
			return None
		else:
			scan_directory(markDir)
			if images or masks:
				return images, masks
			if imagesPath or masksPath:
				return imagesPath, masksPath

	async def ListFiles(self, markDir: str = "") -> Union[List[str], None]:
		images: List[str] = []

		def scan_directory(directory: str):
			for entry in os.scandir(directory):
				if entry.is_dir(follow_symlinks=False):
					scan_directory(entry.path)
				elif entry.is_file(follow_symlinks=False) and entry.path.endswith(
					(".png", ".jpg", ".jpeg", ".webp")
				):
					mark_path = Path(
						entry.path.replace("train", "validation")
					).with_suffix(".png")
					if mark_path.exists():
						images.append(entry.path)
						images.append(str(mark_path))
					else:
						print(f"Faltando: {entry.path.replace('train', 'validation')}")
				else:
					print(f"File Error: {entry.path}")

		if markDir == "":
			print("Nenhum diretório foi repassado!")
		else:
			scan_directory(markDir)
			return images


class TensorLoader:
	@tf.function
	# @profile
	def convertImages(self, image_path: str, mask_path: str):
		img = tf.io.read_file(image_path)
		img = tf.image.decode_png(img, channels=3)  # type: ignore
		img = tf.cast(img, tf.float32) / tf.constant(255, dtype=tf.float32)
		img = tf.image.resize(img, (512, 320), method="nearest")

		mask = tf.io.read_file(mask_path)
		mask = tf.image.decode_png(mask, channels=4)  # type: ignore
		mask = tf.cast(mask, tf.float32) / tf.constant(255, dtype=tf.float32)
		# mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
		mask = tf.image.resize(mask, (512, 320), method="nearest")
		return img, mask

		# @profile
		# @tf.function

	def processImages(imgList):
		# @profile
		def decode_images(imgPath: tuple[str, str]):
			# input_tensor = keras.utils.load_img(path=imgPath, color_mode='rgba')
			# output_tensor = keras.utils.load_img(path=imgPath[1], color_mode='rgba')

			input_bytes = tf.io.read_file(imgPath[0])
			output_bytes = tf.io.read_file(imgPath[1])

			decode_input_1 = (
				tf.image.decode_image(input_bytes, channels=4, dtype=tf.dtypes.float32)
				/ 255.0
			)  # type: ignore
			decode_output_1 = (
				tf.image.decode_image(output_bytes, channels=4, dtype=tf.dtypes.float32)
				/ 255.0
			)  # type: ignore

			# decode_input_2 = tf.image.convert_image_dtype(input_bytes, tf.float32)
			# decode_output_2 = tf.image.convert_image_dtype(output_bytes, tf.float32)

			# <--- Muito Lent0 --->
			# decode_input_3 = np.load(imgPath[0])
			# decode_input_3 = decode_input_3['arr_0']
			# decode_output_3 = np.load(imgPath[1])
			# decode_output_3 = decode_output_3['arr_0']

			# decode_input_4 = tf.cast(input_tensor, tf.float32) / 255.0 # type: ignore
			# decode_output_4 = tf.cast(output_tensor, tf.float32) / 255.0 # type: ignore

			# resize_img = tf.image.resize(decode_img_1, [512 ,320])

			# normalized = tf.cast(decode_img_1, tf.float32) / 255.0 # type: ignore
			# normalized = tf.image.per_image_standardization(decode_img_1)
			# print(tf.reduce_min(decode_img_4), tf.reduce_max(decode_img_4))

			# Optional saving for visualization:
			# keras.preprocessing.image.save_img(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png", normalized)

			# <---- Legacy ---->
			# Teste 1: img_array = tf.cast(tf.clip_by_value(normalized, 0, 1) * 255, tf.uint8).numpy()
			# Teste 2: img_array = (normalized.numpy() * 255).astype(np.uint8)
			# Teste 3: img_array = keras.utils.img_to_array(normalized, dtype='float32')
			# Save: Image.fromarray(img_array, mode="RGBA").save(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png")
			return decode_input_1, decode_output_1

		with tf.device("/CPU:0"):  # type: ignore
			print("Carregando Imagens...")
			decoded_images = [
				tf.expand_dims(decode_images(img), axis=0) for img in tqdm(imgList)
			]
			decoded_images = tf.concat(decoded_images, 0)
			print("Concatenate Terminado...")
			return decoded_images
