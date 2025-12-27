import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer


def preprocess_X(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def preprocess_Y(Y_train, Y_test):
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)
    return Y_train, Y_test, lb


def main():
    data = load_wine()
    X_orig = data.data
    Y_orig = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_orig, Y_orig, test_size=0.3)

    X_train, X_test, scaler = preprocess_X(X_train, X_test)
    Y_train, Y_test, lb = preprocess_Y(Y_train, Y_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=X_train.shape[1]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3)
    ])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )

    model.fit(X_train, Y_train, batch_size=1, epochs=8)

    accuracy = model.evaluate(X_test, Y_test)

if __name__ == "__main__":
    main()
