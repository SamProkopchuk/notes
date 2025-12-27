import sys
sys.path.insert(1, '..')

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from HelperClasses import NPImageDataGenerator

BATCH_SIZE = 128
NUM_CLASSES = 62

# Alleviate memory issues:
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def load_test_data_gen():
    test_batch_gen = tfds.as_numpy(
        tfds.load(
            'emnist',
            split='test',
            # If None -> default path
            data_dir=None,
            batch_size=BATCH_SIZE,
            shuffle_files=False,
            download=False,
            as_supervised=True
        )
    )

    ds_imgen = NPImageDataGenerator(
        test_batch_gen, rescale=1. / 255,
        num_classes=NUM_CLASSES
    )

    return ds_imgen


def main():
    model_paths = (
        './__temp__/models/DFF/uninterpolated',
        './__temp__/models/DFF/interpolated',
        './__temp__/models/CNN/uninterpolated',
        './__temp__/models/CNN/interpolated'
    )

    model_reports = []

    for model_path in model_paths:
        print(f'Evaluating model at \'{model_path}\' model on test set:')
        ds_imgen = load_test_data_gen()
        model = tf.keras.models.load_model(model_path)
        print("Model Summary:", model.summary(), sep="\n")
        loss, accuracy = model.evaluate(ds_imgen)

        model_reports.append(
            f'Model \'{model_path}\' model acheived {accuracy*100:.2f}% accuracy.'
        )

        del ds_imgen
        del model

    for report in model_reports:
        print(report)

if __name__ == "__main__":
    main()
