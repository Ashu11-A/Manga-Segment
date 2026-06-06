"""Dataset-preparation script for U-Net training.

Ports the legacy ``convertImages.py``: resizing dataset images to ``320x512``
RGBA and emitting flipped augmentations, either as compressed ``.npz`` float32
arrays or as ``.png`` files, plus a ``--verify`` mode that flags image/mask
pairs that differ too much. Run directly (``--npz``/``--png``/``--verify``) or
import :func:`resize_images` / :func:`process_files`.

Requires ``tqdm``, ``PIL`` and ``cv2``; reads the ``dados/train`` tree and
writes to ``dados_cache``.
"""

from __future__ import annotations

import asyncio
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from algorithms.unet.data import DataLoader


def resize_images(path: str, type: str) -> None:
	image = Image.open(path)
	image = image.resize((320, 512), Image.LANCZOS)
	image = image.convert("RGBA")

	imageInvert1 = image.transpose(Image.FLIP_LEFT_RIGHT)
	imageInvert2 = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(
		Image.FLIP_TOP_BOTTOM
	)

	# Normalize the float32 data to [0, 1]
	# image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

	# Save the NumPy array as a compressed npz file
	path = str(path).replace("dados", "dados_cache")

	dirName = os.path.dirname(path)
	fileName = os.path.basename(path)
	if not os.path.exists(dirName):
		os.makedirs(dirName)

	# Convert back to 8-bit integer for PNG saving
	# image_array_uint8 = (imageInvert_array * 255).astype(np.uint8)
	# image_png = Image.fromarray(image_array_uint8, mode='RGBA')
	# image_png.save(os.path.join(dirName, fileName.replace(".png", "_float32.png")))

	if type == "npz":
		# Convert the image to a NumPy array with float32 dtype
		image_array = np.array(image, dtype=np.float32) / 255.0
		imageInvert1_array = np.array(imageInvert1, dtype=np.float32) / 255.0
		imageInvert2_array = np.array(imageInvert2, dtype=np.float32) / 255.0
		fileName = fileName.replace((".webp"), ".png")
		np.savez_compressed(
			os.path.join(dirName, fileName.replace((".png"), f".{type}")), image_array
		)
		np.savez_compressed(
			os.path.join(dirName, fileName.replace(".png", f"_invert[0].{type}")),
			imageInvert1_array,
		)
		np.savez_compressed(
			os.path.join(dirName, fileName.replace(".png", f"_invert[1].{type}")),
			imageInvert2_array,
		)
	elif type == "png":
		fileName = fileName.replace((".webp"), ".png")
		image.save(f"{dirName}/{fileName}")

		fileName = fileName.replace(".png", "_invert[0].png")
		imageInvert1.save(f"{dirName}/{fileName}")

		fileName = fileName.replace("_invert[0].png", "_invert[1].png")
		imageInvert2.save(f"{dirName}/{fileName}")


async def process_files(mode: str, path: str = "dados/train") -> None:
	"""Process the dataset tree according to ``mode`` (``npz``/``png``/``verify``)."""
	loader = DataLoader()
	processed = 0
	files = await loader.ListFiles(path)
	if files is not None:
		if mode == "npz":
			for filePath in tqdm(files):
				if filePath.endswith((".png", ".webp")):
					if (
						not os.path.exists(
							filePath.replace("dados", "dados_cache").replace(
								".png", ".npz"
							)
						)
						== True
						and not os.path.exists(
							filePath.replace("dados", "dados_cache").replace(
								".png", "_invert[0].npz"
							)
						)
						== True
						and not os.path.exists(
							filePath.replace("dados", "dados_cache").replace(
								".png", "_invert[1].npz"
							)
						)
						== True
					):
						resize_images(filePath, "npz")
						processed += 1
					else:
						print(f"Já existe: {filePath}")
				else:
					print(f"Arquivo em formato invalido: {filePath}")
		elif mode == "png":
			for filePath in tqdm(files):
				if filePath.endswith((".png", ".webp")):
					if (
						not os.path.exists(filePath.replace("dados", "dados_cache"))
						== True
						and not os.path.exists(
							filePath.replace("dados", "dados_cache").replace(
								".png", "_invert[0].png"
							)
						)
						== True
						and not os.path.exists(
							filePath.replace("dados", "dados_cache").replace(
								".png", "_invert[1].png"
							)
						)
						== True
					):
						resize_images(filePath, "png")
						processed += 1
					else:
						print(f"Já existe: {filePath}")
				else:
					print(f"Arquivo em formato invalido: {filePath}")
		elif mode == "verify":

			def join_strings(array):
				return [list(x) for x in zip(*[iter(list(array))] * 2)]

			files = join_strings(files)
			for filePath in tqdm(files):
				original = cv2.imread(filePath[0], cv2.IMREAD_GRAYSCALE)
				mask = cv2.imread(filePath[1], cv2.IMREAD_GRAYSCALE)
				# Convert the original image to grayscale, if needed.
				if len(original.shape) == 3:
					original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
				if len(mask.shape) == 3:
					mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

				# Resize the mask to the original image size.
				original.astype(np.uint8)
				mask.astype(np.uint8)

				# Absolute difference between the original image and the mask.
				diff = cv2.absdiff(original, mask)
				diff = diff.astype(np.float32)

				# Mean of the difference.
				mean_diff = np.mean(diff)
				if mean_diff > float(50):  # type: ignore
					print(
						f"A máscara e a imagem original são diferentes. ({mean_diff})"
					)
					print(filePath)

		else:
			print("Nenhuma ação selecionada")
		print(f"Imagens processadas: {processed}")
	else:
		print("Nenhum Arquivo Encontrado!")


if __name__ == "__main__":
	import argparse

	parse = argparse.ArgumentParser(
		description="Converter Imagens antes do treinamento"
	)
	parse.add_argument("--npz", action="store_true", help="Converte em Float32")
	parse.add_argument("--png", action="store_true", help="Converte em PNG")
	parse.add_argument("--verify", action="store_true", help="Verifica as imagens")
	args = parse.parse_args()

	if args.npz:
		mode = "npz"
	elif args.png:
		mode = "png"
	elif args.verify:
		mode = "verify"
	else:
		mode = ""

	asyncio.run(process_files(mode))
