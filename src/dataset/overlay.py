"""Desenho de sobreposições visuais dos polígonos detectados.

Usado para a visualização em tempo real: cada página processada é redesenhada
com os polígonos de texto destacados, tanto para salvar uma prévia em disco
quanto para exibir em uma janela ao vivo.
"""

from __future__ import annotations

DEFAULT_OUTLINE_COLOR = (0, 255, 0)  # verde no espaço BGR do OpenCV


def draw_polygons_overlay(
	image_bgr,
	polygons,
	*,
	outline_color: tuple[int, int, int] = DEFAULT_OUTLINE_COLOR,
	fill_alpha: float = 0.35,
	line_thickness: int = 2,
	annotate_index: bool = True,
):
	"""Devolve uma cópia da imagem com os polígonos preenchidos e contornados.

	``polygons`` é uma sequência de matrizes ``(num_pontos, 2)`` em coordenadas
	de pixel. O preenchimento é semitransparente (combinado via
	``cv2.addWeighted``) para que o conteúdo da página continue visível sob as
	máscaras.
	"""
	import cv2
	import numpy as np

	annotated_image = image_bgr.copy()
	if not len(polygons):
		return annotated_image

	# Camada separada apenas para o preenchimento translúcido.
	fill_layer = annotated_image.copy()
	integer_polygons = [
		np.asarray(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
		for polygon_points in polygons
	]
	for polygon_pixels in integer_polygons:
		cv2.fillPoly(fill_layer, [polygon_pixels], outline_color)
	cv2.addWeighted(fill_layer, fill_alpha, annotated_image, 1.0 - fill_alpha, 0.0, annotated_image)

	# Contorno opaco e índice de cada polígono por cima do preenchimento.
	for polygon_index, polygon_pixels in enumerate(integer_polygons):
		cv2.polylines(
			annotated_image,
			[polygon_pixels],
			isClosed=True,
			color=outline_color,
			thickness=line_thickness,
		)
		if annotate_index:
			anchor_point = polygon_pixels.reshape(-1, 2)[0]
			label_position = (int(anchor_point[0]), int(anchor_point[1]) - 5)
			cv2.putText(
				annotated_image,
				str(polygon_index + 1),
				label_position,
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				outline_color,
				1,
				cv2.LINE_AA,
			)
	return annotated_image
