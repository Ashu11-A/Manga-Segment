import keras
from keras import layers
from keras_tuner import HyperParameters

def TrainModel(hp: HyperParameters):
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(768, 512, 4)))
    
    hp_units_1 = hp.Int('units1', min_value=8, max_value=256, step=2, sampling="log")
    model.add(layers.Dense(units=hp_units_1, activation='relu'))

    hp_units_2 = hp.Int('units2', min_value=8, max_value=256, step=2, sampling="log")
    model.add(layers.Dense(units=hp_units_2, activation='relu'))

    hp_units_3 = hp.Int('units3', min_value=8, max_value=256, step=2, sampling="log")
    model.add(layers.Dense(units=hp_units_3, activation='relu'))
    
    # hp_units_end = hp.Int(name='units', min_value=4, max_value=10, step=32)
    model.add(layers.Dense(units=4, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), # type: ignore
        metrics=['accuracy'],
        # run_eagerly=True
    )

    model.summary()

    return model

def LoaderModel():
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(768, 512, 4)))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dense(units=4, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=['accuracy'],
    )

    return model

