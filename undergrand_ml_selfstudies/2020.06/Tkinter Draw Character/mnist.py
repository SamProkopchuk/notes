import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import shift

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def interpolated(X, Y, shift_values=[-4, 0, 4]):
    X_cpy = X.copy()
    Y_cpy = Y.copy()
    for shift_up in shift_values:
        for shift_right in shift_values:
            print(f'Shifting {shift_up}, {shift_right}')
            if shift_up == shift_right == 0: continue
            shifted = shift(X, shift=(0, shift_up, shift_right))
            X_cpy = np.r_[X_cpy, shifted]
            Y_cpy = np.r_[Y_cpy, Y]
    return X_cpy, Y_cpy


def get_data():
    print('Getting data')

    # Check if saved preprocessed data exists:
    try:
        X_train = np.load('./__temp__/data/X_train_interpolated.npy')
        Y_train = np.load('./__temp__/data/Y_train_interpolated.npy')
         
    except FileNotFoundError:

        ((X_train_orig, Y_train_orig), ) = tfds.as_numpy(tfds.load(
            'mnist',
            split=['train'],
            batch_size=-1,
            as_supervised=True
        ))

        print('Preprocessing')

        lb = LabelBinarizer()
        Y_train = lb.fit_transform(Y_train_orig)

        X_train = X_train_orig.reshape(-1, 28, 28)
        X_train, Y_train = interpolated(X_train, Y_train)

        X_train = np.reshape(X_train, (-1, 28 * 28))
        X_train = X_train / 255.0

        Path('./__temp__').mkdir(parents=True, exist_ok=True)
        Path('./__temp__/data').mkdir(parents=True, exist_ok=True)
        Path('./__temp__/pickle').mkdir(parents=True, exist_ok=True)

        lb_file = open('./__temp__/pickle/lb', 'wb')
        pickle.dump(lb, lb_file)
        lb_file.close()

        X_train, Y_train = shuffle(X_train, Y_train)
        

        np.save('./__temp__/data/X_train_interpolated.npy', X_train)
        np.save('./__temp__/data/Y_train_interpolated.npy', Y_train)
    
    else:
        lb_file = open('./__temp__/pickle/lb', 'rb')
        lb = pickle.load(lb_file)
        lb_file.close()
        assert(isinstance(lb, LabelBinarizer))

    # Load the test data with tfds:
    ((X_test_orig, Y_test_orig), ) = tfds.as_numpy(tfds.load(
        'mnist',
        split=['test'],
        batch_size=-1,
        as_supervised=True
    ))

    Y_test = lb.transform(Y_test_orig)

    X_test = np.reshape(X_test_orig, (-1, 28 * 28))
    X_test = X_test / 255.0

    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'lb': lb
    }
    
    return data


def main():
    data = get_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(784),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['categorical_accuracy']
    )

    model.fit(X_train, Y_train, batch_size=512, epochs=20)
    model.save('./__temp__/models/mnist_save1')
    model.evaluate(X_test, Y_test)

if __name__ == '__main__':
    main()
