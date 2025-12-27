import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

WINDOW_SIZE = 7


def preprocessed(data, random_state=None):
    lb = LabelBinarizer()
    data = lb.fit_transform(data)
    X_orig = np.empty((1, WINDOW_SIZE * data.shape[1]))
    Y_orig = np.empty((1, data.shape[1]))
    for windex in range(data.shape[0] - WINDOW_SIZE):  # windex very nice
        X_orig = np.r_[X_orig, data[
            windex:windex + WINDOW_SIZE, :].reshape(1, -1)]
        Y_orig = np.r_[Y_orig, data[
            windex + WINDOW_SIZE, :].reshape(1, -1)]
    X_orig = X_orig[1:, :]
    Y_orig = Y_orig[1:, :]
    return X_orig, Y_orig, lb


def main():
    data = np.genfromtxt("1959.txt", dtype=np.short,
                         skip_header=False).reshape(-1, 1)

    X_orig, Y_orig, lb = preprocessed(data)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_orig, Y_orig, shuffle=False, test_size=0.2)

    X_train, Y_train = shuffle(X_train, Y_train)

    X_dev, X_test, Y_dev, Y_test = train_test_split(
        X_orig, Y_orig, shuffle=True, test_size=0.2)

    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=X_orig.shape[1]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(Y_orig.shape[1])
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['categorical_accuracy']
    )

    model.fit(X_train, Y_train, batch_size=16, epochs=10)

    error, accuracy = model.evaluate(X_dev, Y_dev)
    print(f"{accuracy*100:0.2f}% Accuracy on dev set.")

    if input("Would you like to evaluate the NN on the test set? [Y/n]: ") in ("y", "Y"):
        error, accuracy = model.evaluate(X_test, Y_test)
        print(f"{accuracy*100:0.2f}% Accuracy on test set.")

if __name__ == "__main__":
    main()
