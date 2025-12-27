import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from HelperClasses import NPImageDataGenerator

BATCH_SIZE = 128
EPOCHS = 3

# Total number of target classes
NUM_CLASSES = 62


# Following info aquired from:
# https://www.tensorflow.org/datasets/catalog/emnist
TRAIN_SAMPLES = 697932
TEST_SAMPLES = 116323

# Alleviate memory issues:
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def load_data_gen():
    """
    Loads a numpy batch generator of the emnist dataset, repeating EPOCHS times.
    """
    train_batch_gen = tfds.as_numpy(
        tfds.load(
            'emnist',
            split='train',
            # If None -> default path:
            data_dir=None,
            batch_size=BATCH_SIZE,
            shuffle_files=True,
            # Set download to True if you dont have the dataset and your ok with
            # downloading a couple GBs of data.
            download=False,
            as_supervised=True
        ).repeat(EPOCHS)
    )
    return train_batch_gen


def train_and_save_model(imgen, save_path=None):
    """
    Given the image generator "imgen",
    trains a feed forward deep neural network as show below.
    Then saves to save_path if save_path is not None.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['categorical_accuracy']
    )
    model.fit(
        imgen,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=np.ceil(TRAIN_SAMPLES / BATCH_SIZE)
    )
    if save_path is not None:
        model.save(save_path)


def main():
    train_batch_gen = load_data_gen()
    train_imgen = NPImageDataGenerator(
        train_batch_gen, rescale=1. / 255,
        num_classes=NUM_CLASSES
    )
    train_and_save_model(
        train_imgen, save_path='./__temp__/models/DFF/uninterpolated')

    del train_imgen
    del train_batch_gen

    train_batch_gen = load_data_gen()
    train_imgen = NPImageDataGenerator(
        train_batch_gen, rescale=1. / 255,
        height_shift_range=4, width_shift_range=4, shift_fill=0.,
        num_classes=NUM_CLASSES
    )
    train_and_save_model(
        train_imgen, save_path='./__temp__/models/DFF/interpolated')

if __name__ == '__main__':
    main()
