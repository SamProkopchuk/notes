import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D

AUGS_PER_IMG = 3
RESHAPE_SIZE = [200, 200]
TRAIN_SAMPLES = 23262 * (AUGS_PER_IMG + 1)
NUM_CLASSES = 1
BATCH_SIZE = 32
EPOCHS = 50


def get_model():
    '''
    Return uncompiled CNN for cats_vs_dogs dataset.
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def resize_img(image, label):
    image = tf.image.resize(image, RESHAPE_SIZE, method='nearest')
    return image, label


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255
    return image, label


def main():
    train_ds = tfds.load(
        'cats_vs_dogs',
        split='train',
        data_dir='/media/sam/7C6C-8EE6/Datasets/tensorflow_datasets',
        batch_size=1,
        shuffle_files=True,
        as_supervised=True)

    train_ds = train_ds.map(resize_img).map(preprocess)

    model = get_model()
    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    print(model.summary())

    model.fit(
        train_ds.repeat(EPOCHS),
        epochs=EPOCHS,
        steps_per_epoch=np.ceil(TRAIN_SAMPLES / BATCH_SIZE))

if __name__ == '__main__':
    main()
