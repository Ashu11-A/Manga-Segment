"""U-Net architecture builder shared by training and hyperparameter tuning.

The legacy ``unet/model.py`` carried two near-identical ~190-line functions
(``FindModel`` and ``LoaderModel``) that built the exact same encode/decode
network with different parameter sources. Here the encode/decode structure
lives in a single parametrized :func:`build_unet`; ``FindModel`` (the
keras_tuner hypermodel) and ``LoaderModel`` (the fixed-default builder) are thin
wrappers around it, eliminating the duplication.

Keras is imported at module top because this module is only imported lazily by
the training/tuning code paths (never at ``import algorithms`` time).
"""

from __future__ import annotations

import keras
from keras import layers
from keras.layers import (
	BatchNormalization,
	Conv2D,
	Conv2DTranspose,
	Dropout,
	Input,
	ReLU,
	concatenate,
)


def build_unet(
	input_shape: tuple[int, int, int] = (512, 320, 3),
	*,
	filters: int = 16,
	dropout: float = 0.2,
	kernel_size: int = 3,
	kernel_initializer: str = "he_normal",
	activation: str = "relu",
	activation_end: str = "sigmoid",
	pooling: str = "MaxPooling2D",
	out_channels: int = 4,
	loss: str = "BinaryCrossentropy",
	optimizer: str = "Adam",
	learning_rate: float = 0.001,
):
	"""Build and compile the U-Net used for manga segmentation.

	The encode (``down_block``) and decode (``up_block``) stages are defined once
	here; both legacy builders produced this exact graph. Returns a compiled
	:class:`keras.Model`.
	"""

	def down_block(x, block_filters: int, dropout_prob: float = 0, use_maxpool=True):
		x = Conv2D(
			block_filters,
			kernel_size,
			kernel_initializer=f"{kernel_initializer}",
			padding="same",
		)(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Conv2D(
			block_filters,
			kernel_size,
			kernel_initializer=f"{kernel_initializer}",
			padding="same",
		)(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		if dropout_prob > 0:
			x = Dropout(dropout_prob)(x)
		if use_maxpool:
			return getattr(layers, str(pooling))((2, 2))(x), x
		return x, x

	def up_block(x, y, block_filters: int):
		x = Conv2DTranspose(block_filters, kernel_size, strides=(2, 2), padding="same")(x)
		x = concatenate([x, y], axis=3)
		x = Conv2D(
			block_filters,
			kernel_size,
			activation=activation,
			kernel_initializer=f"{kernel_initializer}",
			padding="same",
		)(x)
		x = Conv2D(
			block_filters,
			kernel_size,
			activation=activation,
			kernel_initializer=f"{kernel_initializer}",
			padding="same",
		)(x)
		return x

	inputs = Input(shape=input_shape)

	# encode
	cblock1 = down_block(inputs, filters)
	cblock2 = down_block(cblock1[0], filters * 2)
	cblock3 = down_block(cblock2[0], filters * 4)
	cblock4 = down_block(cblock3[0], filters * 8, dropout_prob=dropout)

	cblock5 = down_block(
		cblock4[0], filters * 16, use_maxpool=False, dropout_prob=dropout
	)

	# decode
	ublock6 = up_block(cblock5[0], cblock4[1], filters * 8)
	ublock7 = up_block(ublock6, cblock3[1], filters * 4)
	ublock8 = up_block(ublock7, cblock2[1], filters * 2)
	ublock9 = up_block(ublock8, cblock1[1], filters)

	conv9 = Conv2D(
		filters,
		3,
		activation="relu",
		padding="same",
		kernel_initializer=f"{kernel_initializer}",
	)(ublock9)

	conv10 = Conv2D(out_channels, (1, 1), activation=activation_end, dtype="float32")(
		conv9
	)

	model = keras.Model(inputs, conv10, name="u-net")

	model.compile(
		loss=getattr(keras.losses, str(loss))(),
		optimizer=getattr(keras.optimizers, str(optimizer))(learning_rate),
		metrics=["accuracy"],
	)
	model.summary()

	return model


def FindModel(hp):
	"""keras_tuner hypermodel: read the search space, then build via ``build_unet``.

	Every hyperparameter name and value set matches the legacy ``FindModel``.
	"""
	loss = hp.Choice("loss", ["BinaryCrossentropy"])
	optimizer = hp.Choice("optimizer", ["Adam"])
	activation = hp.Choice("activation", ["relu"])
	activation_end = hp.Choice("activation_end", ["sigmoid"])
	pooling = hp.Choice("pooling", ["MaxPooling2D"])
	# upscale = hp.Choice('upscale', ['Conv2DTranspose'])
	learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])
	kernel_initializer = hp.Choice("kernel_initializer", ["he_normal"])
	kernel_size = hp.Choice("kernel_size", values=[3])
	dropout = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
	filter = hp.Choice("filter", values=[4, 8, 16])

	return build_unet(
		(512, 320, 3),
		filters=filter,
		dropout=dropout,
		kernel_size=kernel_size,
		kernel_initializer=kernel_initializer,
		activation=activation,
		activation_end=activation_end,
		pooling=pooling,
		out_channels=4,
		loss=loss,
		optimizer=optimizer,
		learning_rate=learning_rate,
	)


def LoaderModel():
	"""Build the U-Net with the legacy fixed defaults (filter=16, dropout=0.2, ...)."""
	return build_unet()
