import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential

TRAIN_SAMPLES = 60000
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 5


def allow_growth():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


def get_model():
    '''
    Returns uncompiled CNN model which is designed for fashion mnist
    '''
    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, (2, 2), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


def preprocess(inputs, targets):
    inputs = tf.cast(inputs, tf.float32) / 255.
    return inputs, targets


def main():
    allow_growth()
    model = get_model()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

    train_ds, test_ds = tfds.load(
        'fashion_mnist',
        split=['train', 'test'],
        data_dir='/run/media/sam/7C6C-8EE6/Datasets/tensorflow_datasets',
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        download=False,
        as_supervised=True
    )
    train_ds = train_ds.map(
        preprocess
        ).shuffle(
        TRAIN_SAMPLES, reshuffle_each_iteration=True)

    model.fit(
        train_ds.repeat(EPOCHS),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=np.ceil(TRAIN_SAMPLES / BATCH_SIZE)
    )

    test_ds = test_ds.map(preprocess)
    model.evaluate(test_ds)

if __name__ == '__main__':
    main()
