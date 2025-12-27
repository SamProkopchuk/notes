import numpy as np
import os.path
import tensorflow as tf
import tensorflow_datasets as tfds

from itertools import chain
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tfrecord_tools import NumpyTFRecordManager

TRAIN_SAMPLES = 1034 * 4
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 200


def allow_growth():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

def get_augmented_train_ds(file_path: str):
    tfr_manager = NumpyTFRecordManager(
        file_path, compression_type='ZLIB', encoding_type='JPEG')

    if not os.path.exists(file_path):
        def _np_augentation_generator(ds, imgen, augs_per_img):
            '''
            A generator function:
            For each image, yield original image and 
            augs_per_img augmented versions produced by imgen.
            '''
            for image, label in ds:
                orig_and_augs = chain([image], imgen.flow(image, batch_size=1))
                for _ in range(augs_per_img + 1):
                    aug_img = next(orig_and_augs)
                    yield aug_img.astype(image.dtype), label
        train_ds = tfds.as_numpy(
            tfds.load(
                'beans',
                split='train',
                # data_dir='/run/media/sam/7C6C-8EE6/Datasets/tensorflow_datasets/',
                batch_size=1,
                shuffle_files=True,
                download=True,
                as_supervised=True))
        imgen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        train_aug_gen = _np_augentation_generator(train_ds, imgen, augs_per_img=3)
        tfr_manager.iterable_to_tfrecord(train_aug_gen, verbose=True)

    train_ds = tfr_manager.tfrecord_to_tfds(
        output_shapes=[[500, 500, 3], [1]],
        output_dtypes=[tf.uint8, tf.int64])
    return train_ds

def shard_ds(ds, basename_path, num_shards, verbose=True):
    shard_file_paths = []
    for index in range(num_shards):
        ds_shard = ds.shard(num_shards, index)
        shard_file_path = os.path.join(basename_path) + f'-{index:04}-of-{num_shards:04}'
        shard_file_paths.append(shard_file_path)
        if os.path.exists(shard_file_path): continue
        tfr_manager = NumpyTFRecordManager(shard_file_path, compression_type='ZLIB', encoding_type='JPEG')
        tfr_manager.iterable_to_tfrecord(ds_shard.as_numpy_iterator(), verbose=verbose)
    return shard_file_paths

def get_model():
    '''
    Returns uncompiled CNN model which is designed for fashion mnist
    '''
    model = tf.keras.models.Sequential()
    model.add(Conv2D(
        3, (2, 2), strides=(2, 2),
        input_shape=(500, 500, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(
        9, (3, 3), strides=(1, 1), padding='same',
        input_shape=(250, 250, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(
        (2, 2),
        input_shape=(250, 250, 9)))
    model.add(Conv2D(
        27, (5, 5), strides=(3, 3),
        input_shape=(125, 125, 9), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(
        (2, 2),
        input_shape=(41, 41, 27)))
    model.add(Conv2D(
        81, (3, 3), strides=(1, 1), padding='same',
        input_shape=(20, 20, 27), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(
        81, (3, 3), strides=(1, 1), padding='same',
        input_shape=(20, 20, 27), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(
        27, (2, 2), strides=(1, 1),
        input_shape=(20, 20, 128), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(
        (2, 2),
        input_shape=(18, 18, 27)))
    model.add(Flatten(
        input_shape=(9, 9, 27)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


def preprocess(inputs, targets):
    inputs = tf.cast(inputs, tf.float32) / 255.
    return inputs, targets


def main():
    allow_growth()

    save_prefix = './__temp__/' # '/run/media/sam/7C6C-8EE6/Datasets/augmented_image_data/tfds_beans/'
    file_name = 'tfds_beans.tfrecord'
    file_path = os.path.join(save_prefix, file_name)

    train_ds = get_augmented_train_ds(file_path)
    # Now we do a bunch of shaizha to get relatively-shuffled data:
    num_shards = 10
    train_ds = train_ds.shuffle(32)
    shard_file_paths = shard_ds(train_ds, basename_path=file_path, num_shards=num_shards)
    shard_files_ds = tf.data.Dataset.from_tensor_slices(shard_file_paths).shuffle(num_shards)

    # Kinda gross way of just using one method:
    tfr_manager = NumpyTFRecordManager('', compression_type='ZLIB', encoding_type='JPEG')
    train_ds = shard_files_ds.interleave(
        lambda file_path: tfr_manager.tfrecord_to_tfds(
            output_shapes=[[500, 500, 3], [1]],
            output_dtypes=[tf.uint8, tf.int64],
            file_path=file_path),
        cycle_length=num_shards).shuffle(12, reshuffle_each_iteration=True)

    val_ds, test_ds = tfds.load(
        'beans',
        split=['validation', 'test'],
        # data_dir='/run/media/sam/7C6C-8EE6/Datasets/tensorflow_datasets/',
        batch_size=BATCH_SIZE,
        shuffle_files=False,
        download=True,
        as_supervised=True)

    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    model = get_model()
    model.compile(
        optimizer='SGD',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])
    print(model.summary())

    model.fit(
        train_ds.batch(BATCH_SIZE).repeat(EPOCHS),
        epochs=EPOCHS,
        steps_per_epoch=np.ceil(TRAIN_SAMPLES / BATCH_SIZE),
        validation_data=val_ds)

    model.evaluate(test_ds)

if __name__ == '__main__':
    main()
