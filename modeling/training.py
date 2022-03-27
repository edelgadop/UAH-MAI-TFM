from preprocessing import preprocessing
from ingestion import ingestion
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models, layers

MATFILE_IMDB = "./images/imdb_crop/imdb.mat"


def create_cnn_gender():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (4, 4), activation='relu', input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def create_cnn_age():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (4, 4), activation='relu', input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    return model


if __name__ == "__main__":
    df_metadata = ingestion.get_metadata(matfile_imdb=MATFILE_IMDB, matfile_wiki=None)
    input_tensor, gender_labels, age_labels, ids = preprocessing.get_input_tensor("images/imdb_crop/", df_metadata,
                                                                                  img_size=128)
    X = np.expand_dims(input_tensor, axis=3)  # target tensor shape is (M, h, w, 1) - 1 channel images
    y_gender = np.float32(gender_labels.reshape((len(gender_labels), 1)))  # gender labels
    y_age = np.float32(age_labels.reshape((len(age_labels), 1)))  # age labels

    model_gender = create_cnn_gender()
    model_age = create_cnn_age()

    model_gender.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
                         metrics=['accuracy', 'precision', 'recall'])
    model_age.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    history_gend = model_gender.fit(X, y_gender, validation_split=0.2, epochs=5, batch_size=128)
    history_age = model_age.fit(X, y_age, validation_split=0.2, epochs=5, batch_size=128)

    model_gender.save("./models/gender_classifier")
    model_age.save("./models/gender_age")